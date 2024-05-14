import torch
import matplotlib
import matplotlib.pyplot as plt
from DINO.collect_dino_features import *
from DINO.dino_wrapper import *
import cv2
'''
The python wrapper for DINO feature extractor
Requirements: DINO folder to be placed!
'''
def extract_dino_image(image, dino_cfg):
    '''
    The function to extract the dino feature of the whole image
    Input: image: an image file read by opencv
    Return: dino feature of the image in numpy format
    '''
    # If unfortunately, dino_cfg is not specified, build up the default cfg

    if dino_cfg is None: 
        # Configuratino for the dino model
        # Hard-code the default cfg for building DINO model
        cfg = {}
        cfg['dino_strides'] = 4
        cfg['desired_height'] = image.shape[0]
        cfg['desired_width'] = image.shape[1]
        cfg['use_16bit'] = False
        cfg['use_traced_model'] = False
        cfg['cpu'] = False
        cfg['similarity_thresh'] = 0.95

        dino_cfg = cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.hub.set_dir("./DINO/hub")

    # Get the point-wise features
    model = get_dino_pixel_wise_features_model(cfg = dino_cfg, device = device)


    img_feat = preprocess_frame(image, cfg=dino_cfg)
    img_feat = model(img_feat)

    img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
    return img_feat_norm.detach().cpu().numpy()

def extract_dino_point(image, p, dino_cfg):
    '''
    The function to extract the dino feature of the selected point
    Input: image: an image file read by opencv
    Return: dino feature of the selected point in numpy format
    '''
    dino_feature_image = extract_dino_image(image, dino_cfg)
    selected_embedding = dino_feature_image[0, :, p[0], p[1]]  # (512,)
    return selected_embedding

def dino_highest_similarity(image, dino_cfg, dino_feat, verbose = False):
    '''
    The function to find the point with the highest dino similarity
    w.r.t the given dino feature
    Return: the point with the highest similarity
    '''
    dino_feature_image = extract_dino_image(image, dino_cfg)

    # Convert them into pytorch tensor and Move them to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_feature_image = torch.from_numpy(dino_feature_image).to(device, dtype=torch.float32)
    dino_feat = torch.from_numpy(dino_feat).to(device, dtype=torch.float32)

    
    # Calculate the similarity
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    similarity = cosine_similarity(
        dino_feature_image, dino_feat.view(1, -1, 1, 1)
    )
    # Viz thresholded "relative" attention scores
    similarity = (similarity + 1.0) / 2.0  # scale from [-1, 1] to [0, 1]
    # similarity = similarity.clamp(0., 1.)
    similarity_rel = (similarity - similarity.min()) / (
        similarity.max() - similarity.min() + 1e-12
    )
    similarity_rel = similarity_rel[0]  # 1, H // 2, W // 2 -> # H // 2, W // 2
    similarity_rel[similarity_rel < 0.1] = 0.0

    idx = np.argmax(similarity_rel.detach().cpu().numpy())
    rx, ry = (int)(idx / similarity_rel.shape[0]), (int)(idx % similarity_rel.shape[0])
    if verbose: # If verbose, also look at an illustration
        # Construct the image for evaluation
        img_to_viz = copy.deepcopy(image)
        img_to_viz = cv2.cvtColor(img_to_viz, cv2.COLOR_BGR2RGB)
        img_to_viz = cv2.resize(img_to_viz, (dino_feature_image.shape[-1], 
                                            dino_feature_image.shape[-2])
                                )
        cmap = matplotlib.cm.get_cmap("jet")
        similarity_colormap = cmap(similarity_rel.detach().cpu().numpy())[..., :3]

        _overlay = img_to_viz.astype(np.float32) / 255
        _overlay = 0.5 * _overlay + 0.5 * similarity_colormap

        fig = plt.figure()
        plt.imshow(_overlay)
        plt.title("Similarity to current feature")
        plt.scatter([rx], [ry])
        plt.show()

        plt.close("all")
    return [rx, ry]

def point_match_sift(image, image2, pick_point1, verbose = False):
    '''
    The function to find the matched point in image2 using SIFT
    '''
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Conver the two images into grayscale
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good_matches.append([m])

    # Store the keypoints in a numpy array
    keypoints_img1 = []
    keypoints_img2 = []

    for match in good_matches:
        p1 = kp1[match[0].queryIdx].pt
        p2 = kp2[match[0].trainIdx].pt

        keypoints_img1.append(p1)
        keypoints_img2.append(p2)
    

    keypoints_img1 = np.array(keypoints_img1)
    keypoints_img2 = np.array(keypoints_img2)

    # Find the transformation H between the two series of keypoints
    # THe formula used: 
    #  x1, x2, ...     h1, h2, h3   x1', x2', ...
    # [y1, y2, ...] = [h4, h5, h6] [y1', y2', ...]
    #  1, 1, ...        0,  0, 1     1,   1, ...
    
    # Construct the equations
    N = keypoints_img1.shape[0]
    keypoints_img1_homo = np.hstack((keypoints_img1, np.ones((N, 1))))
    lhs_upper = np.hstack((keypoints_img1_homo, np.zeros((N, 3))))
    lhs_lower = np.hstack((np.zeros((N, 3)), keypoints_img1_homo))
    lhs = np.vstack((lhs_upper, lhs_lower))

    rhs = np.hstack((keypoints_img2[:, 0], keypoints_img2[:, 1])).reshape(-1, 1)
    
    print(lhs.shape)
    print(rhs.shape)
    # Find the transformation
    h = np.matmul(np.linalg.pinv(lhs), rhs).flatten()
    h = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]]])
    print(h)
    # Find the pixel of the matched point in the other image
    res = np.matmul(h, np.array([[pick_point1[0]], [pick_point1[1]], [1]])).flatten()
    rx, ry = res
    rx = int(rx)
    ry = int(ry)

    if verbose: # If verbose, illustrate the match
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,\
                                  None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()


        # Illustrate the matched point of the selected point on image2
        cv2.circle(image2, (rx, ry), radius=2, color=(255, 0, 0))
        cv2.imshow('Matched Point on Image 2', image2)

        
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return [rx, ry]