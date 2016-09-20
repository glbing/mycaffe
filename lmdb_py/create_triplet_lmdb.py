#coding=utf-8
from PIL import Image
import lmdb
import numpy
import caffe 
import os
import random

def read_images(images_path,new_height,new_width,shuffle):#图像路径列表
	
	# triplet a,p,b
	a_list=[]
	p_list=[]
	n_list=[]
	labels_list=[]

	f=open(images_path)
	if not f:
		print "can not open %s" %images_path
		return 

	list_=f.readlines(80000)

	if shuffle==True:
		random.shuffle(list_)


	for i in range(len(list_)):
		temp=list_[i].strip().split()# len(temp) = 3,a,p,n
		a_list.append(str(temp[0]))
		#print a_list
		p_list.append(str(temp[1]))
		n_list.append(str(temp[2]))
		###labels_list
		labels_list.append(int(1))

	triplet_count=len(a_list)
	print triplet_count

	images_array=[]
	
	for i in range(triplet_count):

		a_data=numpy.array(Image.open(a_list[i]).resize((new_height,new_width)))#.transpose((2,0,1))# c*h*w
		p_data=numpy.array(Image.open(p_list[i]).resize((new_height,new_width)))#.transpose((2,0,1))# c*h*w 
		n_data=numpy.array(Image.open(n_list[i]).resize((new_height,new_width)))#.transpose((2,0,1))# c*h*w 
		if (len(a_data.shape)+len(p_data.shape)+len(n_data.shape)) == 3*3:
			a_data=a_data.transpose((2,0,1))
			p_data=p_data.transpose((2,0,1))
			n_data=n_data.transpose((2,0,1))

			image_data=numpy.zeros((3*3,new_width,new_width))
		
			image_data[0:3,:,:]=a_data
			image_data[3:6,:,:]=p_data
			image_data[6:9,:,:]=n_data
		
			images_array.append(image_data)

		else:
			del a_list[i]
			del p_list[i]
			del n_list[i]
			del labels_list[i]
		#print image_data.dtype
	return a_list,p_list,n_list,images_array,numpy.array(labels_list)


# lndb_dir:数据库目录   images_array：n*3*h*w    batch_size:每次写入的batch大小
#  writting lmdb is a superposition way
def convert_imageset(lmdb_dir,a_list,p_list,n_list,images_array,images_label,batch_size):

	'''
	if os.listdir(lmdb_dir):
		os.system("rm  %s/*" %lmdb_dir)
		print "rm %s" %lmdb_dir
	'''
	triplet_count=len(a_list)
	print triplet_count

	images_size=images_array[0].nbytes*triplet_count
	map_size=images_size*10

	lmdb_env=lmdb.open(lmdb_dir,map_size)
	lmdb_txn=lmdb_env.begin(write=True)#句柄

	
	for i in range(triplet_count):
		img=images_array[i]
		data=img.astype(numpy.uint8)
		label=images_label[i]
		datum=caffe.proto.caffe_pb2.Datum()
		datum=caffe.io.array_to_datum(data, label)#

		##
		#print datum.channels,datum.height,datum.width

		keystr = '{:0>8d}'.format(i)+'_'+a_list[i]+'_'+p_list[i]+'_'+n_list[i]
		#keystr = int(i)
		lmdb_txn.put(keystr, datum.SerializeToString())#写入内存
		
		if (i+1)%batch_size == 0:
			lmdb_txn.commit()#写入硬盘
			print "batch %d writen" %(i+1)
			lmdb_txn=lmdb_env.begin(write=True)
	# last batch
	lmdb_txn.commit()#写入硬盘	
	lmdb_env.close()



if __name__ == "__main__":
	os.chdir("/home/glb/Consumer-to-shop Clothes Retrieval Benchmark/Img")
	a_list,p_list,n_list,images_array,images_label=read_images("/home/glb/Consumer-to-shop Clothes Retrieval Benchmark/Eval/triplet_train_list.txt",256,256,True)
	convert_imageset("/home/glb/Consumer-to-shop Clothes Retrieval Benchmark/Eval/db_dir",a_list,p_list,n_list,images_array,images_label,100)
	#print images_array
	#print images_array.nbytes
	#print type(images_label[0])
	

