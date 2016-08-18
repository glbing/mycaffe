#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/layers/1bottom_triplet_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template <typename Dtype>
void Triplet1LossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  margin = this->layer_param_.triplet_loss_param().margin();//距离参数a
  // number of triplet in a batch
  num_triplets = this->layer_param_.triplet_loss_param().num_triplets();
  // dimension of each descriptor
  int dim = bottom[0]->count()/bottom[0]->num();//即channels 
  // In each set, we have:
  // the descriptor of reference sample, closest sample, and negative samples
  // number of sets in the whole batch
  int num_set = bottom[0]->num()/(2 + num_triplets);
  CHECK_EQ(bottom[0]->channels(), dim);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  

  

  diff_ap_.Reshape(num_set,dim,1,1);
  diff_an_.Reshape(num_set,dim,1,1);
  diff_pn_.Reshape(num_set,dim,1,1);

  diff_sq_ap_.Reshape(num_set,dim,1,1);
  diff_sq_an_.Reshape(num_set,dim,1,1);

  dist_sq_ap_.Reshape(num_set, 1, 1, 1);
  dist_sq_an_.Reshape(num_set, 1, 1, 1);

/*
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
  */  
}
template <typename Dtype>
void Triplet1LossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  //Dtype margin = this->layer_param_.triplet_loss_param().margin();//距离参数a
  //Dtype losstype = this->layer_param_.triplet_loss_param().losstype();//??
  //int num_triplets = this->layer_param_.triplet_loss_param().num_triplets();
  //int use_pair = this->layer_param_.triplet_loss_param().use_pair();
  CHECK_EQ(bottom[0]->num()%(2 + num_triplets), 0);//模运算
  int dim = bottom[0]->count()/bottom[0]->num();//每一个图片的slice
  int num_set = bottom[0]->num()/(2 + num_triplets);//一个batch中三元组的数目
  //下面求梯度数据
  for (int i=0;i<num_set;++i)
  {
    //diff_ap_
    caffe_sub(dim,bottom[0]->cpu_data()+i*(2+num_triplets)*dim,
      bottom[0]->cpu_data()+((2+num_triplets)*i+1)*dim,diff_ap_.mutable_cpu_data()+i*dim);
    //diff_an_
    caffe_sub(dim,bottom[0]->cpu_data()+i*(2+num_triplets)*dim,
      bottom[0]->cpu_data()+((2+num_triplets)*i+2)*dim,diff_an_.mutable_cpu_data()+i*dim);
    //diff_pn_
    caffe_sub(dim,bottom[0]->cpu_data()+((2+num_triplets)*i+1)*dim,
      bottom[0]->cpu_data()+((2+num_triplets)*i+2)*dim,diff_pn_.mutable_cpu_data()+i*dim);
  }
  Dtype loss(0.0);
  //求ap和an的距离
  for (int i=0;i<num_set;++i)
  {
      dist_sq_ap_.mutable_cpu_data()[i] = caffe_cpu_dot(dim,diff_ap_.cpu_data()+i*dim,
        diff_ap_.cpu_data()+i*dim);//返回内积
      dist_sq_an_.mutable_cpu_data()[i] = caffe_cpu_dot(dim,diff_an_.cpu_data()+i*dim,
        diff_an_.cpu_data()+i*dim);//返回内积   
      Dtype mdist=std::max(margin+dist_sq_ap_.cpu_data()[i]-dist_sq_an_.cpu_data()[i],Dtype(0.0));
      loss += mdist;
      if(mdist==Dtype(0.0))//如果max()==0，就对不会产生loss,也就对backword计算梯度没有贡献
      {
        caffe_set(dim,Dtype(0.0),diff_ap_.mutable_cpu_data()+i*dim);
        caffe_set(dim,Dtype(0.0),diff_an_.mutable_cpu_data()+i*dim);
        caffe_set(dim,Dtype(0.0),diff_pn_.mutable_cpu_data()+i*dim);
      }
  }
  loss = loss / static_cast<Dtype>(num_set) / Dtype(2);//
  top[0]->mutable_cpu_data()[0]=loss;//loss计算完毕
}

template <typename Dtype>
void Triplet1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  //Dtype margin = this->layer_param_.triplet_loss_param().margin();
  //Dtype losstype = this->layer_param_.triplet_loss_param().losstype();
  //int num_triplets = this->layer_param_.triplet_loss_param().num_triplets();
  //int use_pair = this->layer_param_.triplet_loss_param().use_pair();
  if(propagate_down[0])
  {
    int dim = bottom[0]->count()/bottom[0]->num();
    int num_set = bottom[0]->num()/(2 + num_triplets);

    Dtype* diff=bottom[0]->mutable_cpu_diff();
    //bottom[0]:num*dim
    const Dtype alpha=top[0]->cpu_diff()[0]/static_cast<Dtype>(num_set);
    for(int i=0;i<num_set;++i)
    {
      caffe_cpu_axpby(dim,Dtype(-1.0)*alpha,diff_pn_.cpu_data()+i*dim,Dtype(0.0),diff+i*(2+num_triplets)*dim);
      caffe_cpu_axpby(dim,Dtype(-1.0)*alpha,diff_ap_.cpu_data()+i*dim,Dtype(0.0),diff+(i*(2+num_triplets)+1)*dim);
      caffe_cpu_axpby(dim,Dtype(1.0)*alpha,diff_an_.cpu_data()+i*dim,Dtype(0.0),diff+(i*(2+num_triplets)+2)*dim);
    }
  }
}
#ifdef CPU_ONLY
    STUB_GPU(Triplet1LossLayer);
#endif
    INSTANTIATE_CLASS(Triplet1LossLayer);
    REGISTER_LAYER_CLASS(Triplet1Loss);
}  // namespace caffe