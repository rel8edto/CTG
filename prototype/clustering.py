#There's alot of installed modules for this, you will need to grab them to get this to run
# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# for everything else
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
def get_images(path,images):
    os.chdir(path)
    with os.scandir(path) as files:
        for file in files:
            if file.name.endswith('.png'):
                images.append(file)
    return images
# this list holds all the image filename
images=[]
path = r"/home/japan77/Downloads/DSCC383W/CTG/prototype/train_data/images/emergency"
emergency = r"/home/japan77/Downloads/DSCC383W/CTG/prototype/train_data/images/emergency"
religious = r"/home/japan77/Downloads/DSCC383W/CTG/prototype/train_data/images/religious"
images=get_images(path,images)
path=r"/home/japan77/Downloads/DSCC383W/CTG/prototype/train_data/images/religious"
images=get_images(path,images)

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}

# lop through each image in the dataset
for image in images:
    #extract the features and update the dictionary
    feat = extract_features(image,model)
    data[image] = feat
# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)
# reduce the amount of dimensions in the feature vector
maxp=0
maxk=0
maxs=0
for n in range(1,101): #need to update for real runs
    pca = PCA(n_components=n, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)
    print(sum(pca.explained_variance_ratio_))
    for k in range(2,50):
        kmeans=KMeans(n_clusters=k,random_state=22)
        kmeans.fit(x)
        counts=np.bincount(kmeans.labels_)
        if len(np.where(counts>1)[0])!=k: #check to ensure clusters have more than 1 element
            continue
        emergency_counts=[]
        religious_counts=[]
        for i in range(k):
            emergency_counts.append(0)
            religious_counts.append(0)
        for file, cluster in zip(filenames, kmeans.labels_):
            if emergency==os.path.dirname(file):
                emergency_counts[cluster]+=1
            elif religious==os.path.dirname(file):
                religious_counts[cluster]+=1
        emergency_success=0
        religious_success=0
        for i in range(k):
            if emergency_counts[i]>religious_counts[i]:
                emergency_success+=emergency_counts[i]
            if emergency_counts[i]<religious_counts[i]:
                religious_success+=religious_counts[i]
        #these are hardcoded because I'm lazy
        total_success=(emergency_success+religious_success)/2443
        print("Current n,k: %d, %d" %(n,k))
        print("Emergency success %:"+str(emergency_success/1462))
        print("Relgiious success %:"+str(religious_success/981))
        print("Total success %:"+str(total_success))
        if total_success>maxs:
            maxs=total_success
            maxp=n
            maxk=k
print("Maximum n,k,success: %d, %d, %f" %(maxp,maxk,maxs))
# cluster feature vectors
k=2
kmeans = KMeans(n_clusters=k,random_state=22) #as there should only be 2 clusters
kmeans.fit(x)
# holds the cluster id and the images { id: [images] }
groups = {}
y=copy.deepcopy(x)
c=kmeans.cluster_centers_
y=np.append(y,c,axis=0)
projection=TSNE().fit_transform(y)
emergency_counts=[0,0]
religious_counts=[0,0]
col=[]
edge=[]
lab=[]
label=[]
for file, cluster in zip(filenames,kmeans.labels_):
    if emergency==os.path.dirname(file):
        emergency_counts[cluster]+=1
        col.append('g')
        lab.append('Emergency')
        if cluster==1:
            edge.append('m')
            label.append('Cluster 1')
        else:
            edge.append('y')
            label.append('Cluster 0')
    elif religious==os.path.dirname(file):
        religious_counts[cluster]+=1
        col.append('b')
        lab.append('Religious')
        if cluster==1:
            edge.append('m')
            label.append('Cluster 1')
        else:
            edge.append('y')
            label.append('Cluster 0')
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)
for l in range(k):
    col.append('r')
    lab.append('cluster center')
    label.append('cluster center')
    edge.append('r')
visited=set()
plt.figure(figsize=(6,4))
for loc,c,lb in zip(projection,col,lab):
    if lb not in visited:
        visited.add(lb)
        plt.scatter(*loc,c=c,label=lb)
    else:
        plt.scatter(*loc,c=c)
plt.legend()
plt.show()
plt.figure(figsize=(6,4))
visited=set()
for loc,c,lb in zip(projection,edge,label):
    if lb not in visited:
        visited.add(lb)
        plt.scatter(*loc,c=c,label=lb)
    else:
        plt.scatter(*loc,c=c)
plt.legend()
plt.show()
print("emergency:"+str(emergency_counts))
print("religious:"+str(religious_counts))
# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
# this is just incase you want to see which value for k might be the best 
sse = []
list_k = list(range(1, 20))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    
    sse.append(km.inertia_)

# Plot sse against k
print(sse)
plt.figure(figsize=(6,4))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
plt.show()
#DBSCAN coded below here
maxn=0
maxe=0
maxs=0
colors=['r','g','b','c','y','m','k','tab:orange','tab:brown']
plt.figure(figsize=(6,4))
visited=set()
list_eps=list(range(1,50,1))
for n in range(2,11):
    for e in list_eps:
        db=DBSCAN(eps=e,min_samples=n).fit(x)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        if n_clusters_>=2:
            emergency_counts=[]
            religious_counts=[]
            for i in range(n_clusters_):
                emergency_counts.append(0)
                religious_counts.append(0)
            print("Current eps,n: %d, %d" %(e,n))
            print("Estimated number of clusters, noise points: %d, %d" % (n_clusters_,n_noise_))
            for file, cluster in zip(filenames, labels):
                if cluster==-1:
                    continue
                if emergency==os.path.dirname(file):
                    emergency_counts[cluster]+=1
                elif religious==os.path.dirname(file):
                    religious_counts[cluster]+=1
            emergency_success=0
            religious_success=0
            for i in range(n_clusters_):
                if emergency_counts[i]>religious_counts[i]:
                    emergency_success+=emergency_counts[i]
                if emergency_counts[i]<religious_counts[i]:
                    religious_success+=religious_counts[i]
            #these are hardcoded because I'm lazy
            total_success=(emergency_success+religious_success)/2443
            print("Emergency success %:"+str(emergency_success/1462))
            print("Relgiious success %:"+str(religious_success/981))
            print("Total success %:"+str(total_success))
            if str(n) not in visited:
                plt.scatter(e,total_success,label=str(n),c=colors[n-2])
                visited.add(str(n))
            else:
                plt.scatter(e,total_success,c=colors[n-2])
            if total_success>maxs:
                maxs=total_success
                maxn=n
                maxe=e
        else:
            if str(n) not in visited:
                plt.scatter(e,0,label=str(n),c=colors[n-2])
                visited.add(str(n))
            else:
                plt.scatter(e,0,c=colors[n-2])
plt.legend(title="Minimum number of\n elements in a cluster:")
plt.xlabel("Maximum distance between 2 points")
plt.ylabel("Sorting accuracy")
plt.show()
print("Maximum n,e,success: %d, %d, %f" %(maxn,maxe,maxs))
