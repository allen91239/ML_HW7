import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
def rbf(img, img_co):
    ncoord = np.array(img)
    ncoord2 = np.array(img_co)
    C_D = distance.cdist(ncoord, ncoord2, metric='sqeuclidean')
    return np.exp(-(3e-9) * C_D)
def linear(img, img_co):
    img = np.array(img)
    img_co = np.array(img_co)
    return np.dot(img, img_co.T)
def read_img(path):
    data = list()
    data2 = list()
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = subdir + os.sep + filename
            #print(filepath)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.resize(img, (100,100), interpolation = cv2.INTER_AREA)
            data.append(np.array(img).flatten())
            data2.append(np.array(img2).flatten())
    return np.array(data), np.array(data2)
def PCA(data):
    new_data = list()
    average = np.mean(data, axis=0)
    for d in data:
        new_data.append(d-average)
    new_data = np.array(new_data)
    #print(new_data)
    pc = list()
    cov = np.dot(new_data, new_data.T)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_vec /= np.linalg.norm(eig_vec, axis=0)
    idx = eig_val.argsort()[::-1]
    eig_vec = eig_vec[:,idx]  #eigenvector sorted from large eigenvalue to small eigenvalue
    eig_vec = eig_vec.real
    for i in range(25):
        pc.append(eig_vec[i])
    pc = np.array(pc).T
    z = np.dot(new_data.T, pc).astype('float32')
    #print(z.shape)
    img = np.zeros((5*231, 5*195))
    for i in range(5):
        for j in range(5):
            for k in range(231):
                for l in range(195):
                    img[i*231+k][j*195+l] = z[k*195+l][i*5+j]
    #print(img.shape)
    plt.imshow(img, cmap ='gray')
    plt.savefig("pca_eigenface.png")
    plt.close()
    return pc, z
def draw(num, data):
    img = np.zeros((2*231, 5*195))
    for i in range(2):
        for j in range(5):
            for k in range(231):
                for l in range(195):
                    img[i*231+k][j*195+l] = data[num[i*5+j]][k*195+l]
    plt.imshow(img, cmap='gray')
    plt.savefig("pca_original.png")
    plt.close()
def reconstruct(num, data, pc,z, average):
    z = np.dot(z, pc.T) + np.tile(average,(135,1)).T
    #print(z)
    
    #print(pc.shape)
    img = np.zeros((2*231, 5*195))
    for i in range(2):
        for j in range(5):
            for k in range(231):
                for l in range(195):
                    img[i*231+k][j*195+l] = z[k*195+l][num[i*5+j]]
    plt.imshow(img, cmap='gray')
    plt.savefig("pca_reconstructed.png")
    plt.close()
def pca_recognition(test, train, z, avg):

    new_test = np.dot(test,z)
    new_train = np.dot(train,z)
    dist = list()
    for x in new_test:
        for y in new_train:
            dist.append(y-x)
    dist = np.array(dist)
    dd = list()
    for d in dist:
        dd.append(np.linalg.norm(d))
    #dist /= np.linalg.norm(dist)
    dist = np.array(dd)
    #print(dist.shape)
    #print(dist)
    test_label = np.zeros(30)
    prd_label = np.zeros(30)
    for i in range(15):
        test_label[2*i] = i
        test_label[2*i+1] = i
    # (30*135, 25), every one of the 30 test_data compared to 135 train_data
    tmp = np.zeros(135)
    for i in range(30):
        for j in range(135):
            tmp[j] = dist[i*135+j]
        idx = tmp.argsort()[:-1]
        knn = (idx[:5]/9).astype(int) # get the most frequent subject
        #print(knn)
        #print(np.argmax(np.bincount(knn)))
        prd_label[i] = np.argmax(np.bincount(knn))
    print(prd_label)
    print("The pca accuracy is: " + str(len(prd_label[prd_label == test_label])/30))
    #print(new_test.shape)
    #print(new_train.shape)
    #print(z.shape) # 45045, 25
    #print(z)
def LDA(data): # data is 135 * 10000
    #print(np.array(data).shape)
    data = np.array(data)
    avg = np.mean(data, axis = 0)
    mean_vector = list()
    SW = 0
    SB = 0
    for i in range(15):
        subject = list()
        for j in range(9):
            #print(data[i*9+j])
            subject.append(data[i*9+j])
        center = np.mean(np.array(subject), axis = 0)
        mean_vector.append(center)
        for d in subject:
            diff = (d - center).reshape((10000, 1)).astype('float32')
            SW += np.dot(diff, diff.T)
    #print(SW)
    #print(SW.shape)
    mean_vector = np.array(mean_vector)
    #print(mean_vector.shape)
    for i in range(15):
        for c in mean_vector:
            diff = (c - avg).reshape((10000, 1)).astype('float32')
            #print(diff)
            #print(diff.shape)
            SB += 9*np.dot(diff, diff.T)
    #print(SB)
    #print(SB.shape)
    M = np.linalg.pinv(SW) * SB
    eig_val, eig_vec = np.linalg.eig(M)
    pc = list()
    eig_vec /= np.linalg.norm(eig_vec, axis=0)
    idx = eig_val.argsort()[::-1]
    eig_vec = eig_vec[:,idx]  #eigenvector sorted from large eigenvalue to small eigenvalue
    eig_vec = eig_vec.real
    for i in range(25):
        pc.append(eig_vec[i])
    pc = np.array(pc).T
    z = pc
    #print(z.shape)
    img = np.zeros((5*100, 5*100))
    for i in range(5):
        for j in range(5):
            for k in range(100):
                for l in range(100):
                    img[i*100+k][j*100+l] = z[k*100+l][i*5+j]
    #print(img.shape)
    plt.imshow(img, cmap ='gray')
    plt.savefig("lda_fisherface.png")
    plt.close()
    return z
def lda_reconstruct(num, data, z):
    avg = np.mean(data, axis = 0)
    z = np.dot(np.dot(data[num], z), z.T)
    #average = np.tile(avg,(10,1))
    #z = z + average
    #print(z)
    
    #print(pc.shape)
    img = np.zeros((2*100, 5*100))
    for i in range(2):
        for j in range(5):
            for k in range(100):
                for l in range(100):
                    img[i*100+k][j*100+l] = z[i*5+j][k*100+l]
    plt.imshow(img, cmap='gray')
    plt.savefig("lda_reconstructed.png")
    plt.close()
    average = np.tile(avg,(10,1))
    z = z + average
    img = np.zeros((2*100, 5*100))
    for i in range(2):
        for j in range(5):
            for k in range(100):
                for l in range(100):
                    img[i*100+k][j*100+l] = z[i*5+j][k*100+l]
    plt.imshow(img, cmap='gray')
    plt.savefig("lda_reconstructed_with_average.png")
    plt.close()
def lda_recognition(test, train, z): # z (10000, 25)
    #rint("LDA")
    """new_test = list()
    new_train = list()
    for t in test: # (30, 10000)
        t = np.dot(t, z)  
        new_test.append(t)  #new_test shape (30, 25) 
    for t in train:
        t = np.dot(t, z)
        new_train.append(t) # (135, 25)
    new_test = np.array(new_test)
    new_train = np.array(new_train)"""
    new_test = np.dot(test,z)
    new_train = np.dot(train,z)
    dist = list()
    for x in new_test:
        for y in new_train:
            dist.append(y-x)
    dist = np.array(dist)
    dd = list()
    for d in dist:
        dd.append(np.linalg.norm(d))
    #dist /= np.linalg.norm(dist)
    dist = np.array(dd)
    #print(dist.shape)
    #print(dist)
    test_label = np.zeros(30)
    prd_label = np.zeros(30)
    for i in range(15):
        test_label[2*i] = i
        test_label[2*i+1] = i
    # (30*135, 25), every one of the 30 test_data compared to 135 train_data
    tmp = np.zeros(135)
    for i in range(30):
        for j in range(135):
            tmp[j] = dist[i*135+j]
        idx = tmp.argsort()[:-1]
        knn = (idx[:5]/9).astype(int) # get the most frequent subject
        #print(knn)
        #print(np.argmax(np.bincount(knn)))
        prd_label[i] = np.argmax(np.bincount(knn))
    print(prd_label)
    print("The lda accuracy is: " + str(len(prd_label[prd_label == test_label])/30))
def KPCA(train, test):
    one_n = np.ones((135,135))/135

    rbf_ker = rbf(train, train) # 135*135
    lin_ker = linear(train, train)
    rbfk = rbf_ker
    link = lin_ker
    #print(rbfk)
    rbf_ker = rbf_ker - np.dot(one_n, rbf_ker) - np.dot(rbf_ker, one_n) + np.dot(one_n, np.dot(rbf_ker, one_n))
    lin_ker = lin_ker - np.dot(one_n, lin_ker) - np.dot(lin_ker, one_n) + np.dot(one_n, np.dot(lin_ker, one_n))
    #print(rbf_ker)
    rbf_eig_val, rbf_eig_vec = np.linalg.eig(rbf_ker)
    lin_eig_val, lin_eig_vec = np.linalg.eig(lin_ker)

    rbf_eig_vec /= np.linalg.norm(rbf_eig_vec, axis=0)
    idx = rbf_eig_val.argsort()[::-1]
    rbf_eig_vec = rbf_eig_vec[:,idx]  #eigenvector sorted from large eigenvalue to small eigenvalue
    rbf_eig_vec = rbf_eig_vec.real
    rbf_pc = list()
    for i in range(25):
        rbf_pc.append(rbf_eig_vec[i])
    rbf_pc = np.array(rbf_pc).T
    rbf_z = rbf_pc # 135*25
    #print(rbf_z)
    lin_eig_vec /= np.linalg.norm(lin_eig_vec, axis=0)
    idx = lin_eig_val.argsort()[::-1]
    lin_eig_vec = lin_eig_vec[:,idx]  #eigenvector sorted from large eigenvalue to small eigenvalue
    lin_eig_vec = lin_eig_vec.real
    lin_pc = list()
    for i in range(25):
        lin_pc.append(lin_eig_vec[i])
    lin_pc = np.array(lin_pc).T
    lin_z = lin_pc



    #print(lin_z)
    #print(lin_z.shape)
    rk = rbf(test, train)  # 30*135
    lk = linear(test, train)
    #print(rk.shape)
    one_n = np.ones((135,135))/135
    one_nl = np.ones((135,30))/135
    rk = rk - np.dot(one_nl.T, rbfk) - np.dot(rk, one_n) + np.dot(one_nl.T, np.dot(rbfk, one_n))
    lk = lk - np.dot(one_nl.T, link) - np.dot(lk, one_n) + np.dot(one_nl.T, np.dot(link, one_n))
    
    rtrain = np.dot(rbf_ker, rbf_z)
    ltrain = np.dot(lin_ker, lin_z)
    rtest = np.dot(rbf_z.T, rk.T).T # 30*25
    ltest = np.dot(lin_z.T, lk.T).T # 30*25
    #print(rtrain)
    #print(rtrain.shape)
    #print(rtest)
    #print(rtest.shape)
    rdist = list()
    for x in rtest:
        for y in rtrain:
            rdist.append(y-x)
    rdist = np.array(rdist)
    dd = list()
    for d in rdist:
        dd.append(np.linalg.norm(d))
    #dist /= np.linalg.norm(dist)
    rdist = np.array(dd)

    ldist = list()
    for x in ltest:
        for y in ltrain:
            ldist.append(y-x)
    ldist = np.array(ldist)
    dd = list()
    for d in ldist:
        dd.append(np.linalg.norm(d))
    #dist /= np.linalg.norm(dist)
    ldist = np.array(dd)

    test_label = np.zeros(30)
    prd_label = np.zeros(30)
    prd_label2 = np.zeros(30)
    for i in range(15):
        test_label[2*i] = i
        test_label[2*i+1] = i
    # (30*135, 25), every one of the 30 test_data compared to 135 train_data
    tmp = np.zeros(135)
    tmp2 = np.zeros(135)
    for i in range(30):
        for j in range(135):
            tmp[j] = rdist[i*135+j]
            tmp2[j] = ldist[i*135+j]
        idx = tmp.argsort()[:-1]
        idx2 = tmp2.argsort()[:-1]
        knn = (idx[:5]/9).astype(int) # get the most frequent subject
        knn2 = (idx2[:5]/9).astype(int)
        #print(knn)
        #print(np.argmax(np.bincount(knn)))
        prd_label[i] = np.argmax(np.bincount(knn))
        prd_label2[i] = np.argmax(np.bincount(knn2))
    print(prd_label)
    print(prd_label2)
    print("The kpca rbf kernel accuracy is: " + str(len(prd_label[prd_label == test_label])/30))
    print("The kpca linear kernel accuracy is: " + str(len(prd_label2[prd_label2 == test_label])/30))
    
def KLDA(train, test):
    #print(np.array(data).shape)
    data = train
    data = np.array(data)
    avg = np.mean(data, axis = 0)
    mean_vector = list()
    SW = 0
    SB = 0
    for i in range(15):
        subject = list()
        for j in range(9):
            #print(data[i*9+j])
            subject.append(data[i*9+j])
        center = np.mean(np.array(subject), axis = 0)
        mean_vector.append(center)
        for d in subject:
            diff = (d - center).reshape((10000, 1)).astype('float32')
            #SW += np.dot(diff, diff.T)
            SW += rbf(diff, diff)
    #print(SW)
    #print(SW.shape)
    mean_vector = np.array(mean_vector)
    #print(mean_vector.shape)
    for i in range(15):
        for c in mean_vector:
            diff = (c - avg).reshape((10000, 1)).astype('float32')
            #print(diff)
            #print(diff.shape)
            #SB += 9*np.dot(diff, diff.T)
            SB += 9*rbf(diff,diff)
    #print(SB)
    #print(SB.shape)
    M = np.linalg.pinv(SW) * SB
    eig_val, eig_vec = np.linalg.eig(M)
    pc = list()
    eig_vec /= np.linalg.norm(eig_vec, axis=0)
    idx = eig_val.argsort()[::-1]
    eig_vec = eig_vec[:,idx]  #eigenvector sorted from large eigenvalue to small eigenvalue
    eig_vec = eig_vec.real
    for i in range(25):
        pc.append(eig_vec[i])
    pc = np.array(pc).T
    z = pc
    new_test = np.dot(test,z)
    new_train = np.dot(train,z)
    dist = list()
    for x in new_test:
        for y in new_train:
            dist.append(y-x)
    dist = np.array(dist)
    dd = list()
    for d in dist:
        dd.append(np.linalg.norm(d))
    #dist /= np.linalg.norm(dist)
    dist = np.array(dd)
    #print(dist.shape)
    #print(dist)
    test_label = np.zeros(30)
    prd_label = np.zeros(30)
    for i in range(15):
        test_label[2*i] = i
        test_label[2*i+1] = i
    # (30*135, 25), every one of the 30 test_data compared to 135 train_data
    tmp = np.zeros(135)
    for i in range(30):
        for j in range(135):
            tmp[j] = dist[i*135+j]
        idx = tmp.argsort()[:-1]
        knn = (idx[:5]/9).astype(int) # get the most frequent subject
        #print(knn)
        #print(np.argmax(np.bincount(knn)))
        prd_label[i] = np.argmax(np.bincount(knn))
    print(prd_label)
    print("The klda rbf kernel accuracy is: " + str(len(prd_label[prd_label == test_label])/30))

def KLDA_lin(train, test):
    #print(np.array(data).shape)
    data = train
    data = np.array(data)
    avg = np.mean(data, axis = 0)
    mean_vector = list()
    SW = 0
    SB = 0
    for i in range(15):
        subject = list()
        for j in range(9):
            #print(data[i*9+j])
            subject.append(data[i*9+j])
        center = np.mean(np.array(subject), axis = 0)
        mean_vector.append(center)
        for d in subject:
            diff = (d - center).reshape((10000, 1)).astype('float32')
            #SW += np.dot(diff, diff.T)
            SW += linear(diff, diff)
    #print(SW)
    #print(SW.shape)
    mean_vector = np.array(mean_vector)
    #print(mean_vector.shape)
    for i in range(15):
        for c in mean_vector:
            diff = (c - avg).reshape((10000, 1)).astype('float32')
            #print(diff)
            #print(diff.shape)
            #SB += 9*np.dot(diff, diff.T)
            SB += 9*linear(diff,diff)
    #print(SB)
    #print(SB.shape)
    M = np.linalg.pinv(SW) * SB
    eig_val, eig_vec = np.linalg.eig(M)
    pc = list()
    eig_vec /= np.linalg.norm(eig_vec, axis=0)
    idx = eig_val.argsort()[::-1]
    eig_vec = eig_vec[:,idx]  #eigenvector sorted from large eigenvalue to small eigenvalue
    eig_vec = eig_vec.real
    for i in range(25):
        pc.append(eig_vec[i])
    pc = np.array(pc).T
    z = pc
    new_test = np.dot(test,z)
    new_train = np.dot(train,z)
    dist = list()
    for x in new_test:
        for y in new_train:
            dist.append(y-x)
    dist = np.array(dist)
    dd = list()
    for d in dist:
        dd.append(np.linalg.norm(d))
    #dist /= np.linalg.norm(dist)
    dist = np.array(dd)
    #print(dist.shape)
    #print(dist)
    test_label = np.zeros(30)
    prd_label = np.zeros(30)
    for i in range(15):
        test_label[2*i] = i
        test_label[2*i+1] = i
    # (30*135, 25), every one of the 30 test_data compared to 135 train_data
    tmp = np.zeros(135)
    for i in range(30):
        for j in range(135):
            tmp[j] = dist[i*135+j]
        idx = tmp.argsort()[:-1]
        knn = (idx[:5]/9).astype(int) # get the most frequent subject
        #print(knn)
        #print(np.argmax(np.bincount(knn)))
        prd_label[i] = np.argmax(np.bincount(knn))
    print(prd_label)
    print("The klda linear kernel accuracy is: " + str(len(prd_label[prd_label == test_label])/30))

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    test_path = os.path.join(script_dir,f"./ML_HW07/Yale_Face_Database/Testing")
    train_path = os.path.join(script_dir,f"./ML_HW07/Yale_Face_Database/Training")

    test_data, test_data2 = read_img(test_path)
    train_data, train_data2 = read_img(train_path)
    
    average = np.mean(train_data, axis=0)
    pc, z = PCA(train_data)
    pick_ten = np.random.randint(135, size=10)
    draw(pick_ten, train_data)
    reconstruct(pick_ten, train_data, pc, z, average)
    pca_recognition(test_data, train_data, z, average)
    
    z = LDA(train_data2)
    lda_reconstruct(pick_ten, train_data2, z)
    lda_recognition(test_data2, train_data2, z)
    
    
    KPCA(train_data, test_data)
    KLDA(train_data2, test_data2)
    KLDA_lin(train_data2, test_data2)
    