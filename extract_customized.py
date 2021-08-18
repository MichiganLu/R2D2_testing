# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use


import os, pdb
from PIL import Image
import numpy as np
import torch
import glob
import cv2
import timeit
import pandas as pd
from scipy.spatial.transform import Rotation as R

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *

#define intrinsic matrix
mint = np.array([[458.654,0,367.215],[0,457.296,248.375],[0,0,1]]) #remember to change it for different cam

def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1024, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores

def return_keypoints(img_path,args,iscuda,net,detector):
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    img = norm_RGB(img)[None] 
    if iscuda: img = img.cuda()
    
    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector,
        scale_f   = args.scale_f, 
        min_scale = args.min_scale, 
        max_scale = args.max_scale,
        min_size  = args.min_size, 
        max_size  = args.max_size, 
        verbose = True)

    xys = xys.cpu().numpy()
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = scores.argsort()[-args.top_k or None:]
    '''
    outpath = img_path + '.' + args.tag
    print(f"Saving {len(idxs)} keypoints to {outpath}")
    np.savez(open(outpath,'wb'), 
        imsize = (W,H),
        keypoints = xys[idxs], 
        descriptors = desc[idxs], 
        scores = scores[idxs])
    '''
    keypoints = xys[idxs] #dimension is N*3, with the last dimension being the patch diameter
    keypoint_ = keypoints[:,:2] #drop the patch diameter and round to int
    #keypoint_ = list(map(tuple, keypoint_))
    descriptors = desc[idxs]    
    return keypoint_, descriptors

def compute_homography(matched_kp1, matched_kp2):
    #matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    #matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_kp1,
                                    matched_kp2,
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers
    
def extract_keypoints(args):
    iscuda = common.torch_set_gpu(args.gpu)

    # load the network...
    net = load_network(args.model)
    if iscuda: net = net.cuda()

    # create the non-maxima detector
    detector = NonMaxSuppression(
        rel_thr = args.reliability_thr, 
        rep_thr = args.repeatability_thr)

    if args.images != None:
        img_pth = args.images
        imgs = sorted(glob.glob(img_pth+'*'))
        imgs = imgs[:500]
        #imgs = glob.glob(img_pth+'*.ppm')
        dict1 = {'detected_matches':[],'valid_matches':[],'inlier_rate':[],'detection_time':[],
                 'EulerZ_error':[],'EulerY_error':[],'EulerX_error':[],'t1_error':[],'t2_error':[],'t3_error':[]}

        print(f"\nExtracting features for {img_pth}")
        #input ground truth dataframe path here
        if args.gtcsv != '':
            gt_dataframe = pd.read_csv(args.gtcsv,usecols=[i for i in range(1,8)],nrows=500)

        for i in range(0,len(imgs)-5,5):
            #time the detection
            start = timeit.default_timer()
            kp1, desc1 = return_keypoints(imgs[i],args,iscuda,net,detector)
            kp2, desc2 = return_keypoints(imgs[i+1],args,iscuda,net,detector)
            stop = timeit.default_timer()
            dict1['detection_time'].append(stop-start)

            #get ground truth
            if args.gtcsv != '':
                t1 = gt_dataframe.iloc[i, 0:3].to_numpy()
                t2 = gt_dataframe.iloc[i + 1, 0:3].to_numpy()
                gt_t = t2-t1
                quaternion1 = gt_dataframe.iloc[i, 3:8].to_numpy()
                quaternion1_scaler_last = np.array([quaternion1[1], quaternion1[2], quaternion1[3], quaternion1[0]])
                rotation1 = R.from_quat(quaternion1_scaler_last).as_matrix()
                quaternion2 = gt_dataframe.iloc[i + 1, 3:8].to_numpy()
                quaternion2_scaler_last = np.array([quaternion2[1], quaternion2[2], quaternion2[3], quaternion2[0]])
                rotation2 = R.from_quat(quaternion2_scaler_last).as_matrix()
                gt_rotation = rotation2 @ rotation1.T


            img1 = cv2.imread(imgs[i])
            img2 = cv2.imread(imgs[i+1])
            '''
            #find matches with FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(desc1,desc2,k=1)  #return best matches, it is a list of list
            matches = [item for sublist in matches for item in sublist] #unpack list of list
            '''
            bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
            matches = bf.match(desc1,desc2)
            dict1['detected_matches'].append(len(matches))
            matches_idx1 = np.array([k.queryIdx for k in matches])
            m_kp1 = [kp1[idx] for idx in matches_idx1]
            m_kp1 = np.vstack(m_kp1)
            #print(m_kp1.shape)##1
            matches_idx2 = np.array([k.trainIdx for k in matches])
            m_kp2 = [kp2[idx] for idx in matches_idx2]
            m_kp2 = np.vstack(m_kp2)
            H, inliers = compute_homography(m_kp1, m_kp2)
            valid_match = np.sum(inliers)
            dict1['valid_matches'].append(valid_match)
            dict1['inlier_rate'].append(valid_match/len(matches))
            print('**************************************')##1
            print(f"Total matches for match{i}.jpg is {valid_match}, outlier rate is {1-valid_match/len(matches)}")##1
            #print(H)
            # Draw matches
            #matches = np.array(matches)[inliers.astype(bool)].tolist()  #throw away outlier matches
            finalkp1 = m_kp1[inliers.astype(bool)]
            finalkp2 = m_kp2[inliers.astype(bool)]
            #print(finalkp1.shape)##1

            #get correct pose, you will have four sets of solution, only one with both positive z value for two cameras
            num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, mint)
            for j in range(len(Rs)):
                left_projection = mint @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]) #world-coor
                right_projection = mint @ np.concatenate((Rs[j], Ts[j]), axis=1)
                triangulation = cv2.triangulatePoints(left_projection, right_projection, finalkp1[0], finalkp2[0]) #point in world-coor
                triangulation = triangulation/triangulation[3] #make it homogeneous (x,y,z,1)
                if triangulation[2] > 0:  #z is positive
                    point_in_cam2 = np.concatenate((Rs[j], Ts[j]), axis=1) @ triangulation #change to cam2 coordinate
                    if point_in_cam2[2] > 0:  #z is positive
                        rotation = Rs[j]
                        translation = Ts[j].squeeze()
                        break
                else:
                    continue
            #record error
            if args.gtcsv != '':
                delta_rotation = rotation.T @ gt_rotation
                euler_error = R.from_matrix(delta_rotation).as_euler('zyx', degrees=True)
                delta_t = translation - gt_t
                dict1['t1_error'].append(delta_t[0])
                dict1['t2_error'].append(delta_t[1])
                dict1['t3_error'].append(delta_t[2])
                dict1['EulerZ_error'].append(euler_error[0])
                dict1['EulerY_error'].append(euler_error[1])
                dict1['EulerX_error'].append(euler_error[2])


            #plot
            # initialize the output visualization image
            (hA, wA) = img1.shape[:2]     #cv2.imread returns (h,w,c)
            (hB, wB) = img2.shape[:2]
            vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
            vis[0:hA, 0:wA] = img1
            vis[0:hB, wA:] = img2
    
            # loop over the matches
            for p1,p2 in zip(finalkp1,finalkp2):
                # only process the match if the keypoint was successfully
                # matched
                # draw the match
                ptA = (int(p1[0]), int(p1[1]))
                ptB = (int(p2[0]) + wA, int(p2[1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
                cv2.circle(vis,ptA,3,color=(0,0,255))
                cv2.circle(vis,ptB,3,color=(0,0,255))
            cv2.imwrite('./output/'+'match'+str(i)+'.jpg',vis)
            print('outputting matching'+str(i))
        #save csv to dict
        if args.gtcsv != '':
            df1 = pd.DataFrame.from_dict(dict1)
            df1.to_csv('./output/evaluation.csv')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--gtcsv",type=str, default='', help='path to ground truth csv file')
    parser.add_argument("--images", type=str, required=True, help='image directory')
    parser.add_argument("--tag", type=str, default='r2d2', help='output file tag')
    
    parser.add_argument("--top-k", type=int, default=5000, help='number of keypoints')

    parser.add_argument("--scale-f", type=float, default=2**0.25)
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    
    parser.add_argument("--reliability-thr", type=float, default=0.7)
    parser.add_argument("--repeatability-thr", type=float, default=0.7)

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')
    args = parser.parse_args()

    extract_keypoints(args)

