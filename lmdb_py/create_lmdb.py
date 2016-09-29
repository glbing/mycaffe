#coding=utf-8
from PIL import Image
import lmdb
import numpy
import caffe 
import os
import sys
import random
import getopt

def read_images(images_path,new_height,new_width,shuffle):#图像路径列表
	
	images_list=[]
	labels_list=[]

	f=open(images_path)
	if not f:
		print "can not open %s" %images_path
		return 

	list_=f.readlines()

	if shuffle==True:
		random.shuffle(list_)


	for i in range(len(list_)):
		temp=list_[i].strip().split()# len(temp) = 2
		images_list.append(str(temp[0]))
		labels_list.append(int(temp[1]))

	images_count=len(images_list)

	images_array=[]
	
	for i in range(images_count):
		image_data=numpy.array(Image.open(images_list[i]).resize((new_height,new_width))).transpose((2,0,1))# c*h*w 
		images_array.append(image_data)
		#print image_data.dtype
	return images_list,images_array,numpy.array(labels_list)


# lndb_dir:数据库目录   images_array：n*3*h*w    batch_size:每次写入的batch大小
def convert_imageset(lmdb_dir,images_list,images_array,images_label,batch_size):

	if os.listdir(lmdb_dir):
		os.system("rm -f %s/*" %lmdb_dir)
		print "rm %s" %lmdb_dir

	images_count=len(images_array)

	images_size=images_array[0].nbytes*images_count
	map_size=images_size*10

	lmdb_env=lmdb.open(lmdb_dir,map_size)
	lmdb_txn=lmdb_env.begin(write=True)#句柄

	
	for i in range(images_count):
		img=images_array[i]
		data=img.astype(numpy.uint8)
		label=images_label[i]
		datum=caffe.proto.caffe_pb2.Datum()
		datum=caffe.io.array_to_datum(data, label)#

		##
		#print datum.channels,datum.height,datum.width

		keystr = '{:0>8d}'.format(i)+'_'+images_list[i]
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


	#images_list,images_array,images_label=read_images("./train1.txt",256,256,True)
	#convert_imageset("db_dir",images_list,images_array,images_label,100)
	#print images_array
	#print images_array.nbytes
	#print type(images_label[0])

	opts,args=getopt.getopt(sys.argv[1:],"h",["help","input_txt=","lmdb_dir=","new_height=","new_width=","shuffle=","batch_size="])
	print opts
	if opts[0][0]=="--help":
		print "a example of usage:"
		print " python db.py --input_txt=test1.txt --lmdb_dir=db_dir1 \n--new_height=256 --new_width=256 --shuffle=True --batch_size=100"
	else:
		input_txt_=opts[0][1]
		lmdb_dir_=opts[1][1]
		new_height_=opts[2][1]
		new_width_=opts[3][1]
		shuffle_=opts[4][1]
		batch_size_=opts[5][1]

		images_list,images_array,images_label=read_images(str(input_txt_),int(new_height_),int(new_width_),bool(shuffle_))
		convert_imageset(str(lmdb_dir_),images_list,images_array,images_label,int(batch_size_))	
