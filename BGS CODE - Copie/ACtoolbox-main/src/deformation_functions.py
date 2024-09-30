# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:48:54 2024

@author: D74179
"""

import cv2
import numpy as np
from skimage.measure import label,regionprops
import math
from skimage.morphology import skeletonize
import meshpy.triangle as triangle
import scipy
from scipy import signal

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions
    
    
    
def angle_two_vectors(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)

    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)

    angle = np.arccos(dot_product)*180/np.pi
    return angle


def calculation_curve_points(mesh_points, mesh_attr, mesh_tris):
    # Calculation of the map correspondance of the curve edges
    Vp, nb_p, Vg, nb_g = [], [], [], []
    count = 0
    for ii in range(len(mesh_points)):
        if mesh_attr[ii] == 0:
            Vg.append(mesh_points[ii])
            nb_g.append(ii)
        else:
            count += 1

    for ii in range(len(mesh_points)):
        if mesh_attr[ii] == 1:
            Vp.append(mesh_points[ii])
            nb_p.append(ii)
            break

    for cc in range(count):
        ind = nb_p[-1]
        neighbors = np.where(mesh_tris == ind)[0]
        tris_neigh = [mesh_tris[ii] for ii in neighbors]
        tris_neigh = np.reshape(tris_neigh, (3*len(tris_neigh)))
        tri = []
        for tt in tris_neigh:
            if tt not in tri and tt != ind and tt not in nb_p:
                tri.append(tt)
        tri2 = []
        for tt in tri:
            if mesh_attr[tt] == 1:
                tri2.append(tt)
        if len(tri2) == 0:
            break
        dis_points = [distance_points(
            mesh_points[ind][0], mesh_points[ind][1], mesh_points[tt][0], mesh_points[tt][1]) for tt in tri2]

        dis_points = np.argsort(dis_points)
        Vp.append(mesh_points[tri2[dis_points[0]]])
        nb_p.append(tri2[dis_points[0]])

    return Vp, nb_p, Vg, nb_g

def calculation_D(nb_p, Vp):
    D = np.zeros((len(nb_p), len(nb_p)))

    for ii in range(0, len(nb_p)-1):
        vi = Vp[ii]
        vj = Vp[ii+1]
        vj1 = Vp[ii-1]
        sum_D = 0
        sum_D += np.dot(vj-vi, np.transpose(vj-vi))
        sum_D += np.dot(vj1-vi, np.transpose(vj1-vi))
        D[ii][ii] = sum_D

    ii = -1
    vi = Vp[ii]
    vj = Vp[0]
    vj1 = Vp[ii-1]
    sum_D = 0
    sum_D += np.dot(vj-vi, np.transpose(vj-vi))
    sum_D += np.dot(vj1-vi, np.transpose(vj1-vi))
    D[ii][ii] = sum_D

    D = np.linalg.inv(D)

    return D


def calculation_EL_ref(n, m, nb_tot, nb_g, mesh_tris, mesh_points, Vg):
    EL_ref = np.zeros((n, n))

    for nb in range(len(nb_g)):
        nb_ini = nb_g[nb]
        neighbors = np.where(mesh_tris == nb_ini)[0]
        tris_neigh = [mesh_tris[ii] for ii in neighbors]
        tris_neigh = np.reshape(tris_neigh, (3*len(tris_neigh)))
        tri = []
        for tt in tris_neigh:
            if tt not in tri and tt != nb_ini:
                tri.append(tt)
        mesh = [(mesh_points[tt], tt) for tt in tri]

        for ii in range(len(mesh)):
            EL_ref[nb+m][np.where(mesh[ii][1] == nb_tot)
                         [0][0]] = np.linalg.norm(mesh[ii][0]-Vg[nb])
            EL_ref[np.where(mesh[ii][1] == nb_tot)[0][0]][nb +
                                                          m] = np.linalg.norm(mesh[ii][0]-Vg[nb])

    return EL_ref


def calculation_EV(nb_Eg, map_Eg, EL_ref, V):
    EV = np.zeros((nb_Eg, 2))
    for ii in range(nb_Eg):
        i = map_Eg[ii][0]
        j = map_Eg[ii][1]
        EV[ii] = ((EL_ref[i][j])/(np.linalg.norm(V[i]-V[j])))*(V[i]-V[j])
    return EV


def calculation_g(Jg):
    g_ref = np.dot(Jg, np.transpose(Jg))
    return g_ref


def calculation_g_value(V, nb_p):
    sum_g = []
    for ii in range(len(nb_p)-1):
        sum_g.append(V[ii][0]*V[ii+1][1]-V[ii+1][0]*V[ii][1])
    sum_g.append(V[len(nb_p)-1][0]*V[0][1]-V[0][0]*V[len(nb_p)-1][1])
    g_ref = np.abs(0.5*np.sum(sum_g))
    return g_ref


def calculation_H(nb_Eg, n, map_Eg):
    H = np.zeros((nb_Eg, n))
    for ii in range(nb_Eg):
        i = map_Eg[ii][0]
        j = map_Eg[ii][1]
        H[ii][i] = 1
        H[ii][j] = -1
    return H


def calculation_ini_U_C(n, upper_cnt, lower_cnt, upper_cnt_I, lower_cnt_I, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, I1, I2, P1, P2, P1_ind, P2_ind):
    list_ini, list_after = [], []  # For display only, can be removed after development

    C = np.zeros((n, n))
    U = np.zeros((n, 2))
    list_dis = []
    # Fit along upper contour
    
    count_upper =0
    count_lower = 0
    
    for ii in range(0, len(upper_cnt_nb)):
        count_upper+=1
        ind_i = upper_cnt_nb[ii]
        list_ini.append(list(upper_cnt[ii]))
        ind = int(np.round(dis_upper_cnt[ii]*len(upper_cnt_I)))
        U[ind_i] = list(upper_cnt_I[ind])
        # distance_points(U[ind_i][0],U[ind_i][1],U[ind_i-1][0],U[ind_i-1][1])>1:
        if True:

            C[ind_i][ind_i] = 1

            list_after.append(list(upper_cnt_I[ind]))
            list_dis.append(dis_upper_cnt[ii])

  
    list_ini.append(list(P2))
    C[P2_ind][P2_ind] = 1
    U[P2_ind] = list(I2)
    list_after.append(list(I2))

    list_dis = []
    # Fit along lower contour
    for ii in range(0, len(lower_cnt)):
        count_lower+=1
        ind_i = lower_cnt_nb[ii]  # Indice of the point in V_ref
        list_ini.append(list(lower_cnt[ii]))
        C[ind_i][ind_i] = 1
        ind = int(np.round(dis_lower_cnt[ii]*len(lower_cnt_I)))
        U[ind_i] = list(lower_cnt_I[ind])
        list_after.append(list(lower_cnt_I[ind]))
        list_dis.append(dis_lower_cnt[ii])

    list_ini.append(list(P1))
    C[P1_ind][P1_ind] = 1
    U[P1_ind] = list(I1)
    list_after.append(list(I1))
    
    
    return C, U, list_ini, list_after


def calculation_interior_points(mesh_points, mesh_points_l, contours_full_l):
    nb_g, Vg = [], []
    for ii in range(len(mesh_points)):
        if [mesh_points_l[ii]] not in contours_full_l:
            Vg.append(mesh_points[ii])
            nb_g.append(ii)
    return nb_g, Vg


def calculation_jacobian(nb_p, V):

    Jg = np.zeros((2, len(V)))
    for ii in range(1, len(nb_p)-1):
        Jg[0][ii] = V[ii+1][1]-V[ii-1][1]
        Jg[1][ii] = V[ii-1][0]-V[ii+1][0]
    ii = 0
    Jg[0][ii] = V[ii+1][1]-V[len(nb_p)-1][1]
    Jg[1][ii] = V[len(nb_p)-1][0]-V[ii+1][0]

    ii = len(nb_p)-1
    Jg[0][ii] = V[0][1]-V[ii-1][1]
    Jg[1][ii] = V[ii-1][0]-V[0][0]

    return Jg


def calculation_L(m, n, nb_p):
    L = np.zeros((m, n))

    for nb in range(1, len(nb_p)-1):
        L[nb][nb] = 1
        L[nb][nb-1] = -0.5
        L[nb][nb+1] = -0.5
    nb = 0
    L[nb][nb] = 1
    L[nb][len(nb_p)-1] = -0.5
    L[nb][nb+1] = -0.5

    nb = len(nb_p)-1
    L[nb][nb] = 1
    L[nb][nb-1] = -0.5
    L[nb][0] = -0.5

    return L


def calculation_M(n, m, nb_tot, nb_g, mesh_tris, mesh_points):
    M = np.zeros(((n-m), n))
    for nb in range(len(nb_g)):
        nb_ini = nb_g[nb]
        neighbors = np.where(mesh_tris == nb_ini)[0]
        tris_neigh = [mesh_tris[ii] for ii in neighbors]
        tris_neigh = np.reshape(tris_neigh, (3*len(tris_neigh)))
        tri = []
        for tt in tris_neigh:
            if tt not in tri and tt != nb_ini:
                tri.append(tt)
        mesh = [(mesh_points[tt], tt) for tt in tri]
        mesh2 = [mesh_points[tt] for tt in tri]
        num = [tt for tt in tri]
        test_sort = sorted_clockwiseangle(mesh_points[nb_ini], mesh2, num)
        w_sum = 0
        vi = mesh_points[nb_ini]
        for ii in range(len(test_sort)):

            if ii == len(mesh)-2:
                vj = test_sort[ii][0]
                vj1 = test_sort[ii+1][0]
                vj2 = test_sort[0][0]
            if ii == len(mesh)-1:
                vj = test_sort[ii][0]
                vj1 = test_sort[0][0]
                vj2 = test_sort[1][0]
            if ii < len(mesh)-2:
                vj = test_sort[ii][0]
                vj1 = test_sort[ii+1][0]
                vj2 = test_sort[ii+2][0]
            dis_points = distance_points(vi[0], vi[1], vj[0], vj[1])

            vector_ji = [vj[0]-vi[0], vj[1]-vi[1]]
            vector_j1i = [vj1[0]-vi[0], vj1[1]-vi[1]]
            vector_j2i = [vj2[0]-vi[0], vj2[1]-vi[1]]
            alphaj = np.arccos((vector_ji[0]*vector_j1i[0]+vector_ji[1]*vector_j1i[1])/(
                (np.sqrt(vector_ji[0]**2+vector_ji[1]**2))*(np.sqrt(vector_j1i[0]**2+vector_j1i[1]**2))))
            alpha_j1 = np.arccos((vector_j1i[0]*vector_j2i[0]+vector_j1i[1]*vector_j2i[1])/(
                (np.sqrt(vector_j1i[0]**2+vector_j1i[1]**2))*(np.sqrt(vector_j2i[0]**2+vector_j2i[1]**2))))

            # We study the point vi = mesh_points[nb_ini] = Vg[nb] and vj = mesh[ii]
            w = ((np.tan(alphaj/2)+np.tan(alpha_j1/2))/dis_points)
            w_sum += w
            M[nb][np.where(test_sort[ii][1] == nb_tot)[0][0]] = w
        M[nb][nb+m] = -1*w_sum

    return M


def calculation_map_Eg(EL_ref):
    find_Eg = np.where(EL_ref > 0)  # Poits
    nb_Eg = int(len(find_Eg[0])/2)
    map_Eg = []
    for ii in range(len(find_Eg[0])):
        if (find_Eg[0][ii], find_Eg[1][ii]) not in map_Eg and (find_Eg[1][ii], find_Eg[0][ii]) not in map_Eg:
            map_Eg.append((find_Eg[0][ii], find_Eg[1][ii]))
    return map_Eg, nb_Eg


def calculation_sigmaV(m, nb_p, V):
    sigmaV = np.zeros((m, 2))
    for nb in range(0, len(nb_p)-1):
        sigmaV[nb] = V[nb]-(V[nb-1]+V[nb+1])/2
    sigmaV[-1] = V[0]-(V[0]+V[-2])/2
    return sigmaV


def calculation_sigmaV0(m, nb_p, Vp):
    sigmaV0 = np.zeros((m, 2))
    for nb in range(0, len(nb_p)-1):
        sigmaV0[nb] = Vp[nb]-(Vp[nb-1]+Vp[nb+1])*0.5
    sigmaV0[-1] = Vp[-1]-(Vp[-2]+Vp[0])*0.5
    return sigmaV0


def calculation_transform_matrix(nb_p, Vp, Vp_ref, D):
    T = np.zeros((len(nb_p), len(nb_p)))

    for ii in range(0, len(nb_p)-1):
        vi = Vp[ii]
        vj = Vp[ii+1]
        vj1 = Vp[ii-1]

        vi_ref = Vp_ref[ii]
        vj_ref = Vp_ref[ii+1]
        vj1_ref = Vp_ref[ii-1]

        sum_T = 0
        sum_T += np.dot(vj-vi, np.transpose(vj_ref-vi_ref))*D[ii][ii]
        sum_T += np.dot(vj1-vi, np.transpose(vj1_ref-vi_ref))*D[ii][ii]
        T[ii][ii] = sum_T

    ii = -1
    vi = Vp[ii]
    vj = Vp[0]
    vj1 = Vp[ii-1]

    vi_ref = Vp_ref[ii]
    vj_ref = Vp_ref[0]
    vj1_ref = Vp_ref[ii-1]

    sum_T = 0
    sum_T += np.dot(vj-vi, np.transpose(vj_ref-vi_ref))*D[ii][ii]
    sum_T += np.dot(vj1-vi, np.transpose(vj1_ref-vi_ref))*D[ii][ii]
    T[ii][ii] = sum_T

    return T


def contours_calculation(dilation,sens):
    dilation = dilation.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ind_max = np.argmax([len(cnt) for cnt in contours]
                        )  # Keep the longest contours
    # Reshape contours coordinates in 2D array
    V_img = np.reshape(contours[ind_max], (len(contours[ind_max]), 2))

    # Sort contour points according to their distance to each other
    contour_sort = [list(V_img[0])]
    ind_sort = [0]
    for i in range(len(V_img)):
        dis = np.asarray([[distance_points(contour_sort[-1][0], contour_sort[-1][1],
                         V_img[j][0], V_img[j][1]), j] for j in range(len(V_img)) if j not in ind_sort])
        if len(dis) != 0:
            test = np.argmin(dis[:, 0])
            contour_sort.append(list(V_img[int(dis[test][1])]))
            ind_sort.append(int(dis[test][1]))
    contour_sort = np.reshape(contour_sort, (len(contour_sort), 2))
    V_img = contour_sort
   
    # Find extremities of the images (head/ back)
    I1, I2, I1_ind, I2_ind = find_extremities(V_img, 0,sens)
    
    # Split between upper and lower contours
    if I2_ind < I1_ind:
        
        upper_cnt_I = V_img[I2_ind+1:I1_ind]
        lower_cnt_I = np.concatenate(
            (V_img[I1_ind+1:len(V_img)], V_img[:I2_ind]))
    else:
        upper_cnt_I = np.flip(V_img[I1_ind+1:I2_ind], 0)
        lower_cnt_I = np.concatenate(
            (np.flip(V_img[:I1_ind], 0), np.flip(V_img[I2_ind+1:len(V_img)], 0)))
        
    return upper_cnt_I, lower_cnt_I, I1, I2, I1_ind, I2_ind


def distance_points(x1, y1, x2, y2):
    return np.sqrt(((x1-x2)**2) + ((y1-y2)**2))

def deformation_model(g_ref,V, Vp, area_ref, nb_tot, n, m, nb_p, L, M, H, C, D, sigmaV0, U, nb_Eg, map_Eg, EL_ref, dilation,nb_frame):
    # Initialization of some of the variables
    Vp_ref = Vp
    Vk1 = V  # Initialization of the points position vector
    # Initialization of the A block
    A = np.block([[L], [M], [H], [C]])
    Vk = Vk1
    Vp = Vk[:len(nb_p), :]
    # # # Update of the B block matrices
    EV = calculation_EV(nb_Eg, map_Eg, EL_ref, Vk)
    T = calculation_transform_matrix(nb_p, Vp, Vp_ref, D)

    sigmaV = np.dot(T, sigmaV0)
    B = np.block([[sigmaV], [np.zeros((n-m, 2))],
                  [EV], [np.abs(U)]])

    # Calculation of the corresponding G matrix and corresponding new points coordinates Vk1: no conservation of the global area
    G = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))
    Vk1 = np.dot(G, B)
    
    return Vk1, U, C, sigmaV, EV

def deformation_model_ini(V, Vp, area_ref, nb_tot, nb_p, n, m, L, M, H, C, D, sigmaV0, U, nb_Eg, map_Eg, EL_ref,dilation):
    # Initialization of some of the variables
    # Display only, can be removed after development process
    g_list, g_value_supp_list, Earea_list = [], [], []
    Earea2_list = []
    Vp_ref = Vp
    g_value = area_ref+100
    g_value_supp = 1000
    count = 0
    Vk1 = V  # Initialization of the points position vector
    # Initialization of the A block

    A = np.block([[0.8*L], [0.8*M], [0.8*H], [1*C]])
    
    V_list, sigmaV_list, EV_list = [], [], []
    # Iteration for the first fitting
    while count < 5:
        if False: 
            break
        else:
            # Update of the points positions vectors
            Vk = Vk1
            Vp = Vk[:len(nb_p), :]
            # # Update of the B block matrices
            EV = calculation_EV(nb_Eg, map_Eg, EL_ref, Vk)
            T = calculation_transform_matrix(nb_p, Vp, Vp_ref, D)
            if count == 0:
                sigmaV = sigmaV0
            else:
                sigmaV = np.dot(T, sigmaV0)
            B = np.block([[0.8*sigmaV], [np.zeros((n-m, 2))],
                          [0.8*EV], [1*np.abs(U)]])
            # Calculation of the corresponding G matrix and corresponding new points coordinates Vk1: no conservation of the global area
            G = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))
            Vk1 = np.dot(G, B)

            # Update of the characteristics used for fitting evaluation
            testimg = np.zeros(dilation.shape)
            # Drawing of the polygon formed by the deformed model
            testarea_img = cv2.fillPoly(
                testimg, np.int32([Vk1[:len(nb_p)]]), (1, 1, 1))

            # Image of the model area and the image area
            testarea = testarea_img+dilation/255
            # Thresholding to keep only the overlapping area
            _, testarea = cv2.threshold(testarea, 1, 1, cv2.THRESH_BINARY)

            # Length of the overlapping area between the image and the model: 1st criteria of end of deformation iteration
            g_value = np.sum(testarea)
            # Length of the area of the image not part of the overlapping area: 2nd criteria of end of deformation iteration
            g_value_supp = np.sum(testarea_img)-g_value
            area_deformed_model = calculation_g_value(
                Vk1, nb_p)  # Area of the deformed model
            
            if g_value<area_deformed_model:
                Earea = (area_ref*area_deformed_model)/(g_value)**2
            else:
                Earea = 3
            
         
            if Earea-1 >= 0:
                Earea_list.append(Earea-1)
            else:
                Earea2_list.append(Earea-1)

            # Display only, can be removed after development process
            g_list.append(np.abs(g_value-area_ref))
            # Display only, can be removed after development process
            g_value_supp_list.append(g_value_supp)
            count += 1  # Iteration count
            V_list.append(Vk1)
            h_test = 0
            sigmaV_list.append(sigmaV)
            EV_list.append(EV)

    if len(Earea_list)!=0:
        ind_it = np.argmin(Earea_list)
        Vk1 = V_list[ind_it]
        sigmaV = sigmaV_list[ind_it]
        EV = EV_list[ind_it]
    else:        
        ind_it = np.argmax(Earea2_list)
        Vk1 = V_list[ind_it]
        sigmaV = sigmaV_list[ind_it]
        EV = EV_list[ind_it]

    Vplot = np.zeros((n,2))
    for ii in nb_tot:
        Vplot[nb_tot[ii]] = Vk1[ii]
    
    return Vk1, g_list, g_value_supp_list, U, C, sigmaV, EV, h_test

def deformation_model_normal(P1_ind,P2_ind,upper_cnt_nb, lower_cnt_nb, test_img,Vk1, V, Vp, area_ref, nb_p, nb_tot, n, m, L, M, H, C, D, sigmaV0, U, nb_Eg, map_Eg, EL_ref):
    # Initialization of some of the variables
    area_list = []
    V_list, px_list = [], []
    V_ref = Vk1.copy()
    U = Vk1
    test_img_inv = test_img
    Vk = V_ref
    Vp_ref = V_ref[:len(nb_p)]
    list_it =[]
    it = 0  # Pourra etre supprimer à la fin de développement
    pas = 0.25    
    right, left, top, bottom = find_extremities_tot(np.asarray(Vk[:len(Vp)]),0)
    
    while it <= 5:  
          # Step for each iteration
        # Update of the U and C matrices for deformation along the normal lines
        U, C = update_U_C_normal_deformation(P1_ind,P2_ind,upper_cnt_nb, lower_cnt_nb, nb_p, pas, Vk, Vp, U, C, it)

        # Update of the A block
        A = np.block([[0.1*L], [0.1*M], [0.1*H], [1*C]])

        # Update of the Vp matrix
        Vp = Vk[:len(nb_p), :]

        # Update of the B block matrices
        EV = calculation_EV(nb_Eg, map_Eg, EL_ref, Vk)
        T = calculation_transform_matrix(nb_p, Vp, Vp_ref, D)
        if it == 0:
            sigmaV = sigmaV0
        else:
            sigmaV = np.dot(T, sigmaV0)
        B = np.block([[0.1*sigmaV], [np.zeros((n-m, 2))],
                     [0.1*EV], [1*np.abs(U)]])

        # Calculation of the corresponding G matrix and corresponding new points coordinates Vk1: no conservation of the global area
        G = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))
        Vk1 = np.dot(G, B)

        if True:#calculation_g_value(Vk1, nb_p)<= area_of_ref:
            testimg = np.zeros(test_img.shape)
            # Drawing of the polygon formed by the deformed model
            testarea = cv2.fillPoly(
                testimg, np.int32([Vk1[:len(nb_p)]]), (1, 1, 1))
            # Intensity of the corresponding area of the polygon
            testarea = testarea*test_img_inv
            area_tot = np.sum(testarea)  # Sum of all pixel intensities of the area
            # For display only, can be removed after development process
            area_list.append(area_tot/calculation_g_value(Vk1, nb_p))
            sum_px_points = []
            mean_area = area_tot/calculation_g_value(Vk1, nb_p)
            for ii in range(len(Vk1)):
                sum_px_points.append(
                    (test_img_inv[int(np.round(Vk1[ii][1]))][int(np.round(Vk1[ii][0]))]-mean_area)**2)
    
            sum_px_points = np.mean(sum_px_points)
            # For display only, can be removed after development process
            px_list.append(sum_px_points/calculation_g_value(Vk1, nb_p))
            # If criteria of end is meet
            if True:  
                Vk = Vk1
            V_list.append(Vk1)
            list_it.append(it)
        it += 1

    if len(px_list)>0:
        ind_it = np.argmax(px_list)
        Vk1 = V_list[ind_it]
    else:
        Vk1 = V_ref

    return Vk1, area_list, px_list, sigmaV, EV


def find_extremities(c, globalOrientation,sens):
    if globalOrientation >= 45:
        right = tuple(c[c[:, 1].argmax()])
        left = tuple(c[c[:, 1].argmin()])
        ind_right = c[:, 1].argmax()
        ind_left = c[:, 1].argmin()
    else:
        ind_max = np.where(c[:, 0] == min(c[:, 0]))
        c_i = [c[ind] for ind in ind_max][0]
        # x_max = c_i[int(len(c_i)/2)]
        ind_f = int(len(c_i)/2)
        ind_left = ind_max[0][ind_f]
        left = tuple(c[ind_left])

        ind_max = np.where(c[:, 0] == max(c[:, 0]))
        c_i = [c[ind] for ind in ind_max][0]
        # x_max = c_i[int(len(c_i)/2)]
        ind_f = int(len(c_i)/2)
        ind_right = ind_max[0][ind_f]
        right = tuple(c[ind_right])

    if sens =='right-to-left':
        return right, left, ind_right, ind_left
    if sens=='left-to-right':
        return left, right, ind_left, ind_right


def find_extremities_tot(c, globalOrientation):
    if globalOrientation >= 45:
        right = tuple(c[c[:, 1].argmax()])
        left = tuple(c[c[:, 1].argmin()])
        bottom = tuple(c[c[:, 0].argmax()])
        top = tuple(c[c[:, 0].argmin()])
    else:
        bottom = tuple(c[c[:, 1].argmax()])
        top = tuple(c[c[:, 1].argmin()])
        right = tuple(c[c[:, 0].argmax()])
        left = tuple(c[c[:, 0].argmin()])

    return right, left, top, bottom

def initialization_model(img,sens):
    # Pre-treatment of the template
    dilation_img = img
    
    reg = regionprops(label(dilation_img))
    ind_max = np.argmax([len(rr.coords) for rr in reg])
    centroid_template = reg[ind_max].centroid
    
    # Calculation of the contours
    contours, hierarchy = cv2.findContours(
        dilation_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # With approximation
    ind_max = np.argmax([len(cnts) for cnts in contours])
    contours = contours[ind_max]  # Keep the longest contour
   
    contours_full, hierarchy = cv2.findContours(
        dilation_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # With all points
    ind_max = np.argmax([len(cnts) for cnts in contours_full])
    contours_full = contours_full[ind_max]
 
    # Reshaping of the vectors of points coordinates
    points = []

    for ii in range(len(contours)):#np.linspace(0, len(contours)-1,np.min((len(contours),100))):  # range(0,len(contours)):#
        ii = int(np.round(ii))
        points.append((contours[ii][0][0], contours[ii][0][1]))

    points_2 = []
    N = 50
    contours_full_reshape =np.reshape(contours_full,(len(contours_full),2))
    P1, P2, P1_ind, P2_ind = find_extremities(contours_full_reshape, 0,sens)
    if P2_ind < P1_ind:
        upper_cnt_tot = contours_full_reshape[P2_ind+1:P1_ind]
    else:
        upper_cnt_tot = np.flip(contours_full_reshape[P1_ind+1:P2_ind], axis=0)

    if P2_ind < P1_ind:
        lower_cnt_tot = np.concatenate((contours_full_reshape[P1_ind+1:len(contours_full_reshape)], contours_full_reshape[:P2_ind]))
    else:
        if P2_ind != len(contours_full_reshape):
            lower_cnt_tot = np.flip(np.concatenate(
                (contours_full_reshape[P2_ind+1:len(contours_full_reshape)], contours_full_reshape[:P1_ind])), axis=0)
        else:
            lower_cnt_tot = np.flip(contours_full_reshape[:P1_ind])

    if P2_ind < P1_ind:
        upper_cnt_nb_tot = np.arange(P2_ind+1, P1_ind)
        lower_cnt_nb_tot = np.concatenate(
            (np.arange(P1_ind+1, len(contours_full_reshape)), np.arange(0, P2_ind)))
    else:
        upper_cnt_nb_tot = np.flip(np.arange(P1_ind+1, P2_ind), axis=0)
        lower_cnt_nb_tot = np.flip(np.concatenate(
            (np.arange(P2_ind+1, len(contours_full_reshape)), np.arange(0, P1_ind))), 0)

    N = 25
    ind_list = []
    for ii in np.linspace(0,len(upper_cnt_tot)-1,N):
        ii = int(np.round(ii))
        ind_list.append(upper_cnt_nb_tot[ii])
    for ii in np.linspace(0,len(lower_cnt_tot)-1,N):
        ii = int(np.round(ii))
        ind_list.append(lower_cnt_nb_tot[ii])
    ind_list = sorted(ind_list)

        
    for ii in ind_list:
        ii = int(np.round(ii))
        points_2.append((contours_full[ii][0][0], contours_full[ii][0][1]))
    
    points = points_2

    # Creation of the meshgrid
    facets = round_trip_connect(0, len(points)-1)
    circ_start = len(points)
    points.extend((3 * np.cos(angle), 3 * np.sin(angle))
                  for angle in np.linspace(0, 2*np.pi, 30, endpoint=False))
    facets.extend(round_trip_connect(circ_start, len(points)-1))
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_holes([(0, 0)])
    info.set_facets(facets)
    mesh = triangle.build(
        info, refinement_func=needs_refinement, generate_faces=True)
    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    mesh_attr = np.array(mesh.point_markers)
    mesh_points_l = mesh_points.copy()
    mesh_points_l = mesh_points_l.astype('uint32')
    mesh_points_l = mesh_points_l.tolist()
    n = len(mesh_points)
    
    # Calculation of the map correspondance of the curve edges
    Vp, nb_p, Vg, nb_g = calculation_curve_points(
        mesh_points, mesh_attr, mesh_tris)
    m = len(Vp)
 
    # Initialization of the vectors of points
    V = np.concatenate((Vp, Vg))

    nb_tot = np.concatenate((nb_p, nb_g))
    V_ref = V.copy()
    # Calculation of the extremities of the template (head/back)
    P1, P2, P1_ind, P2_ind = find_extremities(V_ref, 0,sens)

    # Differenciation between upper and lower points of the contour
    if P2_ind < P1_ind:
        upper_cnt = Vp[P2_ind+1:P1_ind]
        
    else:
        upper_cnt = np.flip(Vp[P1_ind+1:P2_ind], axis=0)

    if P2_ind < P1_ind:
        if P1_ind+1<len(Vp):
            lower_cnt = np.concatenate((Vp[P1_ind+1:len(Vp)], Vp[:P2_ind]))
        else:
            lower_cnt = Vp[:P2_ind]
    else:
        if P2_ind+1< len(Vp):
            lower_cnt = np.flip(np.concatenate(
                (Vp[P2_ind+1:len(Vp)], Vp[:P1_ind])), axis=0)
        else:
            lower_cnt = np.flip(Vp[:P1_ind])

    if P2_ind < P1_ind:
        upper_cnt_nb = np.arange(P2_ind+1, P1_ind)
        # np.concatenate((nb_tot[P1_ind+1:len(Vp)],nb_tot[:P2_ind]))
        if P1_ind+1<len(Vp):
            lower_cnt_nb = np.concatenate(
                (np.arange(P1_ind+1, len(Vp)), np.arange(0, P2_ind)))
        else:
            lower_cnt_nb =  np.arange(0, P2_ind)
    else:
        upper_cnt_nb = np.flip(np.arange(P1_ind+1, P2_ind), axis=0)
        # np.concatenate((nb_tot[P1_ind+1:len(Vp)],nb_tot[:P2_ind]))
        
        lower_cnt_nb = np.flip(np.concatenate(
            (np.arange(P2_ind+1, len(Vp)), np.arange(0, P1_ind))), 0)

    # Calculation of the ratio length of the upper and lower countour at each point
    dis_upper_cnt = ratio_length_pos_contour(upper_cnt, P1, P2)
    dis_lower_cnt = ratio_length_pos_contour(lower_cnt, P2, P1)
    return mesh_points, mesh_tris, nb_p, nb_g, nb_tot, Vp, Vg, V, n, m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind, centroid_template


def mean_value_coords(vi, vj, vj1, vj2):
    alphaj = angle_two_vectors(
        [vj[0]-vi[0], vj[1]-vi[1]], [vj1[0]-vi[0], vj1[1]-vi[1]])
    alphaj1 = angle_two_vectors(
        [vj1[0]-vi[0], vj1[1]-vi[1]], [vj2[0]-vi[0], vj2[1]-vi[1]])
    w = (math.tan(alphaj/2)+math.tan(alphaj1/2))/np.linalg.norm(vi-vj)
    return w


def needs_refinement(vertices, area):
    max_area = 50
    return bool(area > max_area)


def ratio_length_pos_contour(cnt, P1, P2):
    dis_cnt = [distance_points(P2[0], P2[1], cnt[0][0], cnt[0][1])]
    for ii in range(len(cnt)-1):
        dis_cnt.append(distance_points(
            cnt[ii][0], cnt[ii][1], cnt[ii+1][0], cnt[ii+1][1])+dis_cnt[-1])
    dis_cnt.append(distance_points(
        cnt[len(cnt)-1][0], cnt[len(cnt)-1][1], P1[0], P1[1])+dis_cnt[-1])
    dis_tot = dis_cnt[-1]
    dis_cnt = [ii/dis_tot for ii in dis_cnt]
    return dis_cnt


def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]


def sorted_clockwiseangle(origin, points, num):
    origin_bis = [origin[0]+10, origin[1]]

    vector_origin = [origin_bis[0]-origin[0], origin_bis[1]-origin_bis[1]]

    angle_list = np.zeros(len(points))

    for ii in range(len(points)):
        vector = [points[ii][0]-origin[0], points[ii][1]-origin[1]]
        if points[ii][1] > origin[1]:
            angle = 360-np.arccos((vector_origin[0]*vector[0]+vector_origin[1]*vector[1])/((np.sqrt(
                vector_origin[0]**2+vector_origin[1]**2))*(np.sqrt(vector[0]**2+vector[1]**2))))*180/np.pi
        else:
            angle = np.arccos((vector_origin[0]*vector[0]+vector_origin[1]*vector[1])/((np.sqrt(
                vector_origin[0]**2+vector_origin[1]**2))*(np.sqrt(vector[0]**2+vector[1]**2))))*180/np.pi
        angle_list[ii] = angle

    test = np.argsort(angle_list)
    points_sort = [[points[ii], num[ii]] for ii in test]

    return points_sort

def update_mesh_points(Vk1, nb_tot):
    mesh_points = np.zeros((len(nb_tot), 2))
    for ii in range(len(nb_tot)):
        mesh_points[nb_tot[ii]] = Vk1[ii]
    return mesh_points

def update_U_C_normal_deformation(P1_ind,P2_ind,upper_cnt_nb, lower_cnt_nb,nb_p, pas, Vk, Vp, U, C, it):
    betha_list_upper = []
    betha_list_lower = []
    betha_tot= []
    
    for i in range(0, len(nb_p)):  # For each point of contour
        # Calculation of the angle phi corresponding to the angle between the tangent line and the main axis (vertical or horizontal)
        if i == 0:
            phi = math.atan((Vk[len(nb_p)-1][1]-Vk[i+1][1]) /
                            (Vk[len(nb_p)-1][0]-Vk[i+1][0]))
        if i == len(nb_p)-1:
            phi = math.atan((Vk[i-1][1]-Vk[0][1])/(Vk[0][0]-Vk[i+1][0]))
        else:
            phi = math.atan((Vk[i-1][1]-Vk[i+1][1])/(Vk[i-1][0]-Vk[i+1][0]))
        phi = phi*180/np.pi
        betha = np.sign(phi)*(90-np.abs(phi))
        # betha = 90-phi
        betha_tot.append(betha)
        
        if i in upper_cnt_nb :
            x_t = Vk[i][0]+pas*math.cos(betha/180*np.pi)
            y_t = Vk[i][1]-pas*math.sin(betha/180*np.pi)
            if phi<0:
                x_t = Vk[i][0]-pas*math.cos(betha/180*np.pi)
                y_t = Vk[i][1]+pas*math.sin(betha/180*np.pi)
            betha_list_upper.append(betha)
        if i in lower_cnt_nb or i ==P1_ind or i ==P2_ind:
            x_t = Vk[i][0]-pas*math.cos(betha/180*np.pi)
            y_t = Vk[i][1]+pas*math.sin(betha/180*np.pi)
            if phi<0:
                x_t = Vk[i][0]+pas*math.cos(betha/180*np.pi)
                y_t = Vk[i][1]-pas*math.sin(betha/180*np.pi)
            betha_list_lower.append(betha)
       
        U[i][0] = x_t
        U[i][1] = y_t
        C[i][i] = 1
   
    return U, C

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def calculation_tan(nb_p, Vk0):
    tan_list = np.zeros((len(nb_p),1))
    for ii in range(len(nb_p)-1):
        if Vk0[ii+1][0]-Vk0[ii][0] ==0:
            tan_list[ii][0] = ((Vk0[ii+1][1]-Vk0[ii][1])/np.abs(Vk0[ii+1][1]-Vk0[ii][1]))*90
        else:
            tan_list[ii][0] = math.atan((Vk0[ii+1][1]-Vk0[ii][1])/(Vk0[ii+1][0]-Vk0[ii][0]))*180/np.pi

    if Vk0[0][0]-Vk0[-1][0]==0:
        tan_list[-1][0] = ((Vk0[0][1]-Vk0[-1][1])/np.abs(Vk0[0][1]-Vk0[-1][1]))*90
    else:
        tan_list[-1][0] = (Vk0[0][1]-Vk0[-1][1])/(Vk0[0][0]-Vk0[-1][0])

    return tan_list
    

def sorted_V(Vk1,lower_cnt_nb,upper_cnt_nb, P1_ind,P2_ind,dim):
    V_sorted = np.zeros((len(Vk1),len(lower_cnt_nb)+len(upper_cnt_nb)+2,dim))
    
    for v in range(len(V_sorted)):
        count = 0
        for vv in upper_cnt_nb:
            V_sorted[v][count] = Vk1[v][vv]
            # V_sorted[v][count][1] = Vk1[v][vv][1]
            count+=1
        V_sorted[v][count] = Vk1[v][P1_ind]
        count+=1
        for vv in lower_cnt_nb:#[::-1]:
            V_sorted[v][count] = Vk1[v][vv]
            # V_sorted[v][count][1] = Vk1[v][vv][1]
            count+=1
        V_sorted[v][count] = Vk1[v][P2_ind]
    return V_sorted

def model_ellipse_initilization(length,shape,sens,major_axis_length, minor_axis_length):#nb_frame_model,specie,number,V, Vp, area_ref, nb_p, L, M, H, C, D, sigmaV0, U, nb_Eg, map_Eg, EL_ref,n, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind):

    ###############################################################################
    ################### Initialization of the model Ellipse #######################
    
    # Drawing of the ellipse
    img = np.zeros(shape)#((300,300))
    centroid_e = (int(shape[1]/2),int(shape[0]/2))#(150,150)
    major_axis_e = int(major_axis_length/2)#int(150/2) 
    minor_axis_e = int(minor_axis_length/2) #10
    angle_e = 0
    img = cv2.ellipse(img,centroid_e,(major_axis_e,minor_axis_e),angle_e,0,360,1,-1)
    img = img.astype(np.uint8)
    
    # Initialization of the ellipse model
    mesh_points, mesh_tris, nb_p, nb_g, nb_tot, Vp, Vg, V, n, m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind, centroid_template = initialization_model(
        img,sens)

    # Calculation of reference edge lengths
    EL_ref = calculation_EL_ref(n,m, nb_tot, nb_g, mesh_tris, mesh_points, Vg)
    # Calculation of the map correspondance of the inside points edges
    map_Eg, nb_Eg = calculation_map_Eg(EL_ref)
    
    # Calculation of the map correspondance of the curve edges
    L = calculation_L(m, n, nb_p)
    H = calculation_H(nb_Eg, n, map_Eg)  # Calculation of the H matrix
    # Calculation of reference edge lengths
    EL_ref = calculation_EL_ref(n,m, nb_tot, nb_g, mesh_tris, mesh_points, Vg)
    # Calculation of the mean value coordinates
    M = calculation_M(n, m, nb_tot,nb_g, mesh_tris, mesh_points)
    D = calculation_D(nb_p, Vp)  # Calculation of the D matrix
    
    # Calculation of the initialized sigmaV matrix
    sigmaV0 = calculation_sigmaV0(m, nb_p, Vp)

    return V, Vp,centroid_template,mesh_tris, nb_p, nb_tot, L, M, H, D, sigmaV0, nb_Eg, map_Eg, EL_ref,n,m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind


def model_o_initilialization(candidate,sens,mesh_tris,nb_frame_o,V, Vp,centroid_template, nb_p, nb_tot, L, M, H, D, sigmaV0, nb_Eg, map_Eg, EL_ref,n,m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind):
    
    ###########################################################################
    ###################### Treatment of the image chosen ######################
    
    # Reading of the image
    test_img = candidate.image[candidate.indice_img]
    original = candidate.image[candidate.indice_img]
    dilation = cv2.dilate(test_img, np.ones((5,5)))#╠5,10
    
    # Thresholding of the image
    _, dilation = cv2.threshold(dilation, 45, 255, cv2.THRESH_BINARY)
    
    region = regionprops(label(dilation))
    ind_max = np.argmax([len(rr.coords) for rr in region])

    # Calculation of the matrix of rotation
    reg = regionprops(label(dilation))
    ind_max = np.argmax([len(rr.coords) for rr in reg])
    centroid_o = [reg[ind_max].centroid[1], reg[ind_max].centroid[0]]
    # centroid_list.append(centroid_o)
    
    orien_img = reg[ind_max].orientation*180/np.pi # Orientation of the object
    width,height = dilation.shape
    angle = (orien_img/np.abs(orien_img))*(90-np.abs(orien_img)) # Angle of rotation for the object to be aligned with the horizontal axis
    
    # Image rotation to fit the horizontal axis
    rot_mat = cv2.getRotationMatrix2D(center = centroid_o, angle = angle, scale = 1)
    test_img = cv2.warpAffine(test_img,M=rot_mat,dsize=(height,width)) # For binary image
    dilation = cv2.warpAffine(dilation,M=rot_mat,dsize=(height,width)) # For gray scale image  
    
    reg = regionprops(label(dilation))
    ind_max = np.argmax([reg.area for reg in reg])
    # centroid_img = reg[ind_max].centroid
    right, left, top, bottom = find_extremities_tot(reg[ind_max].coords,0)
    centroid_img = [left[0]+0.5*(right[0]-left[0]),top[1]+0.5*(bottom[1]-top[1])]
    ty,tx = centroid_template[0]-centroid_img[0], centroid_template[1]-centroid_img[1]    
    translation_matrix = np.array([[1, 0, tx],[0, 1, ty]], dtype=np.float32)    
    # Translation to fit the centroid of the model
    test_img = cv2.warpAffine(src=test_img, M=translation_matrix, dsize=(height,width)) # For binary image
    _, thresh = cv2.threshold(test_img, 150, 255, cv2.THRESH_BINARY) # Thresholding of the image
    dilation = cv2.dilate(thresh, np.ones((5,5))) # Dilation of the threshold image 5,5
    dilation = dilation.astype(np.uint8)
    # Removing of little objects nearby
    region = regionprops(label(dilation))
    ind_max = np.argmax([len(reg.coords) for reg in region])
    for jj in range(len(region)):
        if jj != ind_max:
            for cc in region[jj].coords:
                dilation[cc[0]][cc[1]] = 0
    
    # Calculation of the area of reference of the studied image
    area_ref = np.sum(dilation/255)

    ###########################################################################
    ##### Calculation of the fixed matrices of the deformation energy term ####

    # Calculation of the jacobian of the polygon area function
    Jg = calculation_jacobian(nb_p, V)
    g_ref = calculation_g(Jg)  # Calculation of the polygon area
    D = calculation_D(nb_p, Vp)  # Calculation of the D matrix Check if it can be removed !!!!!!!


    upper_cnt_I, lower_cnt_I, I1, I2, I1_ind, I2_ind = contours_calculation(
        dilation,sens)
    
    # Initialization of the C and U matrices
    C, U, list_ini, list_after = calculation_ini_U_C(
        n, upper_cnt, lower_cnt, upper_cnt_I, lower_cnt_I, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, I1, I2, P1, P2, P1_ind, P2_ind)
    
    # Calculation of the initialized sigmaV (((matrix
    sigmaV0 = calculation_sigmaV0(m, nb_p, Vp)

    ###########################################################################
    ############ Deformation of the ellipse to fit the image model ############

    # First deformation
    Vk1, g_list, g_value_supp_list, U, C, sigmaV, EV, h_test = deformation_model_ini(V, Vp, area_ref, nb_tot, nb_p, n, m, L, M, H, C, D, sigmaV0, U, nb_Eg, map_Eg, EL_ref,dilation)
    Vp = Vk1[:len(nb_p), :]
    pts = np.array(Vp,np.int32)
    pts = pts.reshape((-1,1,2))
    img_ini = np.zeros(original.shape,dtype=np.uint8)
    cv2.fillPoly(img_ini,[pts],1)
    
    ###########################################################################
    ##################### Initialization of the model 0 #######################
    
    mesh_points, mesh_tris, nb_p, nb_g, nb_tot, Vp, Vg, V, n, m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind, centroid_template = initialization_model(
        img_ini,sens)
    V_ref = V.copy()
    sigmaV0 = calculation_sigmaV0(m, nb_p, Vp)       
    P1, P2, P1_ind, P2_ind = find_extremities(V_ref, 0,sens)
        
    # Calculation of reference edge lengths
    EL_ref = calculation_EL_ref(n, m, nb_tot, nb_g, mesh_tris, mesh_points, Vg)
    # Calculation of the map correspondance of the inside points edges
    map_Eg, nb_Eg = calculation_map_Eg(EL_ref)
    
    # Calculation of the map correspondance of the curve edges
    L = calculation_L(m, n, nb_p)
    H = calculation_H(nb_Eg, n, map_Eg)  # Calculation of the H matrix
    # Calculation of reference edge lengths
    EL_ref = calculation_EL_ref(n, m, nb_tot, nb_g, mesh_tris, mesh_points, Vg)
    # Calculation of the mean value coordinates
    M = calculation_M(n, m, nb_tot, nb_g, mesh_tris, mesh_points)

    V_ref = V.copy()
    Vk1 = V
    C = np.zeros((n, n))
    U = np.zeros((n, 2))
        
    return g_ref, mesh_tris, V, Vp,centroid_template, nb_p, nb_tot, L, M, H, D, sigmaV0, nb_Eg, map_Eg, EL_ref,n,m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind


def calculation_candidate_deformation(candidate):
        
    V, Vp,centroid_template, mesh_tris, nb_p, nb_tot, L, M, H, D, sigmaV0, nb_Eg, map_Eg, EL_ref,n,m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind = model_ellipse_initilization(candidate.length,candidate.img_shape,candidate.sens,candidate.major_axis_length,candidate.minor_axis_length)

    Vk_list, V_tot_list = [], []
    centroid_list, angle_list_tot = [], [] # dilation_list For display only, can be removed after development process
    tan_list_tot = []

    centroid_img_list = []

    g_ref, mesh_tris, V, Vp,centroid_template, nb_p, nb_tot, L, M, H, D, sigmaV0, nb_Eg, map_Eg, EL_ref,n,m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind = model_o_initilialization(candidate,candidate.sens,mesh_tris,candidate.nb_frame_o,V, Vp,centroid_template, nb_p, nb_tot, L, M, H, D, sigmaV0, nb_Eg, map_Eg, EL_ref,n,m, upper_cnt, lower_cnt, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, P1, P2, P1_ind, P2_ind)
    angle_theta_k_list = []
    centroid_k_list = []

    Jg = calculation_jacobian(nb_p, V)
    g_ref = calculation_g(Jg)*5 # Calculation of the polygon area
    D = calculation_D(nb_p, Vp)  # Calculation of the D matrix Check if it can be removed !!!!!!!
    centroid_traj = []
    for ii in range(len(candidate.centroids)):
        img_binary = candidate.image[ii]
        skeleton_test =  skeletonize(img_binary/255)
        for vv in range(len(skeleton_test)):
            for uu in range(len(skeleton_test[0])):
                if skeleton_test[vv][uu] == False:
                    skeleton_test[vv][uu]=0
                else:
                    skeleton_test[vv][uu]=1
                    
        length_fish = np.sum(skeleton_test)
        reg = regionprops(label(skeleton_test))
        ind_max = np.argmax([len(rr.coords) for rr in reg])
        skeleton_coords = reg[ind_max].coords
        skeleton_coords = skeleton_coords[np.argsort(skeleton_coords[:,1])]
        ind_mil = int(len(skeleton_coords)/2)
        if candidate.sens=='right-to-left':
            ind_ter = int(len(skeleton_coords)/3) # indice of the center of the skeleton
        else:
            ind_ter= int((2/3)*len(skeleton_coords)) # indice of the center of the skeleton
        N = 4*int(len(skeleton_coords)/10)
        if candidate.sens=='right-to-left':
            angle_theta_k = math.atan((skeleton_coords[ind_ter][0]-skeleton_coords[ind_mil-N][0])/(skeleton_coords[ind_ter][1]-skeleton_coords[ind_mil-N][1]))*180/np.pi
        else:
            angle_theta_k = math.atan((skeleton_coords[ind_mil+N][0]-skeleton_coords[ind_ter][0])/(skeleton_coords[ind_mil+N][1]-skeleton_coords[ind_ter][1]))*180/np.pi
        angle_theta_k_list.append(angle_theta_k)
        centroid_k_list.append([float(skeleton_coords[ind_ter][1]),float(skeleton_coords[ind_ter][0])])
        
            
        # Calculation of the matrix of rotation
        reg = regionprops(label(img_binary))
        ind_max = np.argmax([len(rr.coords) for rr in reg])
        centroid_o = [reg[ind_max].centroid[1], reg[ind_max].centroid[0]]
        centroid_list.append(centroid_o)
        centroid_traj.append(centroid_o)
        
        orien_img = reg[ind_max].orientation*180/np.pi # Orientation of the object
        width,height = img_binary.shape
        angle = (orien_img/np.abs(orien_img))*(90-np.abs(orien_img)) # Angle of rotation for the object to be aligned with the horizontal axis
        angle_list_tot.append(angle)
        
        # Image rotation to fit the horizontal axis
        rot_mat = cv2.getRotationMatrix2D(center = centroid_o, angle = angle, scale = 1)
        dilation = cv2.warpAffine(img_binary,M=rot_mat,dsize=(height,width)) # For gray scale image
        
        
        
        reg = regionprops(label(dilation))
        ind_max = np.argmax([len(rr.coords) for rr in reg])
        right, left, top, bottom = find_extremities_tot(reg[ind_max].coords,0)
        
        # Calculation of the translation matrix
        reg = regionprops(label(dilation))
        ind_max = np.argmax([reg.area for reg in reg])
        centroid_img = reg[ind_max].centroid
        
        centroid_img_list.append(centroid_img)
        ty,tx = centroid_template[0]-centroid_img[0], centroid_template[1]-centroid_img[1]
        area_ref = np.sum(dilation/255)
        
        ###########################################################################
        ############################ Deformation process ##########################
        
        # Calculation of the jacobian of the polygon area function
        D = calculation_D(nb_p, Vp)  # Calculation of the D matrix
        
        upper_cnt_I, lower_cnt_I, I1, I2, I1_ind, I2_ind = contours_calculation(
            dilation,candidate.sens)
        
        
        
        # Initialization of the C and U matrices
        C, U, list_ini, list_after = calculation_ini_U_C(
            n, upper_cnt, lower_cnt, upper_cnt_I, lower_cnt_I, upper_cnt_nb, lower_cnt_nb, dis_upper_cnt, dis_lower_cnt, I1, I2, P1, P2, P1_ind, P2_ind)
        
        
        
        # Calculation of the initialized sigmaV matrix
        sigmaV0 = calculation_sigmaV0(m, nb_p, Vp)
        
        # First deformation
        Vk1, U, C, sigmaV, EV = deformation_model(g_ref,V, Vp, area_ref, nb_tot, n, m, nb_p, L, M, H, C, D, sigmaV0, U, nb_Eg, map_Eg, EL_ref, dilation,candidate.tot_frame[ii])
        
        
        right, left, top, bottom = find_extremities_tot(Vk1,0)
        
        # Rotaton and translation back to the origin
        rot_mat = cv2.getRotationMatrix2D(center = centroid_o, angle = -angle, scale = 1)
        for vv in range(len(Vk1)):
            Vk1[vv][0] -= tx
            Vk1[vv][1] -= ty
        for vv in range(len(Vk1)):
            Vk1[vv][0] = Vk1[vv][0]*rot_mat[0][0] + Vk1[vv][1]*rot_mat[0][1] + rot_mat[0][2]
            Vk1[vv][1] = Vk1[vv][0]*rot_mat[1][0] + Vk1[vv][1]*rot_mat[1][1] + rot_mat[1][2]
            
        # Update of the variables
        Vk_list.append(Vk1)
        Vp = Vk1[:len(nb_p), :]        
        
        if ii!=0 :                    
            Vk2 = Vk_list[-2].copy()
        
            Vktest = Vk1.copy()
            Vk0 = Vktest.copy()
            
            # Rotation along the trajectory with the previous detection
            theta_trajectory = np.mean([angle_list_tot[-2],angle_list_tot[-1]])
            
            rot_mat = cv2.getRotationMatrix2D(center = centroid_list[-2], angle = theta_trajectory, scale = 1) 
            
            # Rotation of the model at t = t
            for vv in range(len(Vk1)):
                Vk0[vv][0] = Vktest[vv][0]*rot_mat[0][0] + Vktest[vv][1]*rot_mat[0][1] + rot_mat[0][2]
                Vk0[vv][1] = Vktest[vv][0]*rot_mat[1][0] + Vktest[vv][1]*rot_mat[1][1] + rot_mat[1][2]
           
            # Rotation of the model at t = t-1
            
            rot_mat = cv2.getRotationMatrix2D(center = centroid_list[-2], angle = theta_trajectory, scale = 1) 
            # rot_mat = cv2.getRotationMatrix2D(center = centroid_k_list[-2], angle = angle_theta_k_list[-2], scale = 1) 
            
            for vv in range(len(Vk1)):
                Vk2[vv][0] = Vk2[vv][0]*rot_mat[0][0] + Vk2[vv][1]*rot_mat[0][1] + rot_mat[0][2]
                Vk2[vv][1] = Vk2[vv][0]*rot_mat[1][0] + Vk2[vv][1]*rot_mat[1][1] + rot_mat[1][2]
                
                
            # Difference between centroid of both following detections
            right, left, top, bottom = find_extremities_tot(Vk0,0)
            centroid_Vk0 = [left[0]+0.5*(right[0]-left[0]),top[1]+0.5*(bottom[1]-top[1])]
            # test= right[0]
          
            right, left, top, bottom = find_extremities_tot(Vk2,0)
            centroid_Vk2 = [left[0]+0.5*(right[0]-left[0]),top[1]+0.5*(bottom[1]-top[1])]
        
            
            # Translation to fit centroids
            for vv in range(len(Vk0)):
                Vk0[vv][0] += np.round(centroid_Vk2[0]-centroid_Vk0[0]) #-= test#
                Vk0[vv][1] += np.round(centroid_Vk2[1]-centroid_Vk0[1])#(centroid_k_list[-2][1]-centroid_k_list[-1][1])#
                
           
            # Fitting of both detection the most accurately - based on overlapping area
            img_ref = np.zeros(candidate.img_shape,dtype=np.uint8)
            img_ref = cv2.fillPoly(img_ref, np.int32([Vk2[:len(nb_p)]]), 1)
            
            
            test_mean = []
            # test_mean_graph = []
            
            for rr in  np.arange(-5,5.1,0.5):
                Vkmean = Vk0.copy()
                for vv in range(len(Vkmean)):
                    Vkmean[vv][1] += rr
                img_inter = np.zeros(candidate.img_shape,dtype=np.uint8)
                img_inter = cv2.fillPoly(img_inter, np.int32([Vkmean[:len(nb_p)]]), 1)
                
                img_inter = img_inter+img_ref
                _,img_inter = cv2.threshold(img_inter,1,1,cv2.THRESH_BINARY)
                if candidate.sens == 'left-to-right':
                    test_mean.append(np.sum(img_inter[:,int(centroid_Vk2[0]):]))
                else:
                    test_mean.append(np.sum(img_inter[:,:int(centroid_Vk2[0])]))
        
            translation_y = np.arange(-5,5.1,0.5)[np.argmax(test_mean)]#ind_test_mean[np.argmax(test_mean)]
          
            for vv in range(len(Vk0)):
                Vk0[vv][1] += translation_y#centroid_list[-2][1]-centroid_list[-1][1]#centroid_list[-2][0]-centroid_list[-1][0]
        
        
             
            # plt.figure()
            # plt.imshow(dilation, cmap='gray')
            # Vplot = np.zeros((n, 2))
            # for ii in nb_tot:
            #     Vplot[nb_tot[ii]] = [Vk2[ii][0],Vk2[ii][1]]
            
            # plt.triplot(Vplot[:, 0], Vplot[:, 1], mesh_tris,color='green')
            Vplot = np.zeros((n, 2))
            for ii in nb_tot:
                Vplot[nb_tot[ii]] = [Vk0[ii][0],Vk0[ii][1]]
            # plt.triplot(Vplot[:, 0], Vplot[:, 1], mesh_tris,color='orange')
            
        
            # plt.show()
            # plt.title(nb_frame)
            
            # plt.savefig(
            #     'C:/Users/d74179/Documents/Documents/Etude_Body_Deformation/Test/'+str(nb_frame)+'_2.png')
            # # plt.close('all')
        
            # img_tosave = cv2.imread(
            #     'C:/Users/d74179/Documents/Documents/Etude_Body_Deformation/Test/'+str(nb_frame)+'_2.png')
            # gif_list.append(img_tosave)
            # Recording of the gif of deformation (sur-place)
            # Calculation of the tangent vector
            sigma_test = calculation_tan(nb_p, Vk0)#tan_list_tot[-1]-calculation_tan(nb_p, Vk0)
            # sigma_test = tan_list_tot[-1]-sigma_test
            tan_list_tot.append(sigma_test)
            
            # Calculation of the deformation vector
            h_test = (Vk0-Vk2)/length_fish*100 #[(Vk0[pp][1]-Vk2[pp][1]) for pp in range(len(Vp))] #[distance_points(Vk0[pp][0],Vk0[pp][1],Vk2[pp][0],Vk2[pp][1]) for pp in range(len(Vp))]
            
            V_tot_list.append(h_test) 

    V_tot_list= sorted_V(V_tot_list,lower_cnt_nb,upper_cnt_nb,P1_ind,P2_ind,2)

    # tan_list_tot = sorted_V(tan_list_tot,lower_cnt_nb,upper_cnt_nb,P1_ind,P2_ind,1)
        
    # Calculation of the deformation vector along the y axis (= main axis)
    Vtot_list1 = []
    # Vtot_list1 = np.reshape(Vtot_list1,(Vtot_list1.shape[0],Vtot_list1.shape[1]))
    for ee in range(len(V_tot_list)):
        Vtot_list1.append(V_tot_list[ee][:len(nb_p),1])
    Vtot_list1 = np.asarray(Vtot_list1)
    
    return Vtot_list1

def analysis_candidate_deformation(candidate,deformation_matrix,FPS_rate):
    
    matrix_corr_linear_gauss = []
    # Interpolation linéaire
    max_val = deformation_matrix.shape[0] ## same as N
    N = deformation_matrix.shape[0] ## same as max_val
    Ni=FPS_rate*max_val

    dx = max_val/N ## always 1
    x = np.arange(0,max_val,dx) ## wouldn't it be just a 0, as max_val and dx are the same and np,aragne excludes the stop ?
    # x_plot= [FPS_rate*jj for jj in range(max_val+1)]

    # dxi = dx*len(x)/Ni
    # xi = np.arange(0,max_val,dxi)
    
    x = np.linspace(0,len(deformation_matrix)-1,num=len(deformation_matrix),endpoint=True)
    xnew = np.linspace(0,len(deformation_matrix)-1,num=Ni,endpoint=True)
    for ii in range(deformation_matrix.shape[1]):
        f = scipy.interpolate.interp1d(x,deformation_matrix[:,ii],kind='linear')
        inter = np.asarray(f(xnew))
        if ii==0:
            f_interp=np.asarray(inter[:,np.newaxis])
        else:
            f_interp = np.concatenate((f_interp,np.asarray(inter[:,np.newaxis])),axis=1)

    # plt.figure()
    # plt.imshow(f_interp)
    # plt.title('f_interp')
# plt.savefig('C:/Users/D74179/Documents/Documents/Etude_Body_Deformation/Analysis/Intercorrelation/Global/Linear_interpolation')     
    
    gaussian = [math.exp(-((x-24)**2)/(2*((5)**2))) for x in range(len(f_interp[0]))]
    # plt.figure(figsize=(15,15))
    for n in range(len(f_interp)):
        # amplitude_list.append(max(np.abs(matrix_corr_linear[n])))
        gaussian = [(1/max(np.abs(f_interp[n])))*math.exp(-((x-24)**2)/(2*((1.5)**2))) for x in range(len(f_interp[0]))]
    
        matrix_corr_linear_gauss.append(signal.correlate(f_interp[n],gaussian))

    
        
    # plt.savefig('C:/Users/D74179/Documents/Documents/Etude_Body_Deformation/Analysis/Intercorrelation/Global/Linear_gaussian')   
    # plt.figure()
    # plt.imshow(matrix_corr_linear_gauss,cmap='viridis')


    u_tot,v_tot,s_tot=[],[],[]
    # plt.figure()
    
    # pca = PCA(n_components=1, svd_solver='full')
    u,s,v = scipy.linalg.svd(matrix_corr_linear_gauss, full_matrices=False)
    u_tot.append(np.round(u[:, :2],2))
    v_tot.append(np.round(v[:2, :],2))
    s_tot.append(s)
    # comps = [100, 1, 2, 3, 4, 5]
    var_explained = np.round(s**2/np.sum(s**2), decimals=6)
    reconstruction = u[:, :2] @ np.diag(s[:2]) @ v[:2, :]
    
    # plt.imshow(reconstruction,cmap='viridis')
    # plt.title('PCA')
    candidate.add_deformation(reconstruction)

    
    return candidate
