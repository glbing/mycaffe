
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe { 


 /**
 * @brief Computes the Tripletloss
 */
template <typename Dtype>
class TripletLossLayer : public LossLayer<Dtype> {
public:
 explicit TripletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param){}
virtual void LayerSetUp(const vector<Blob<Dtype>*>&bottom,
      const vector<Blob<Dtype>*>& top);
 
virtual inline int ExactNumBottomBlobs() const { return 4; }//返回bottom的数量
virtual inline const char* type() const { return "TripletLoss";}//返回layer的类型
/**
  * Unlike most loss layers, in the TripletLossLayer we can backpropagate反向传播
  * to the first three inputs.
*/
virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 3;
}
 
protected:
virtual void Forward_cpu(const vector<Blob<Dtype>*>&bottom,
      const vector<Blob<Dtype>*>& top);
virtual void Forward_gpu(const vector<Blob<Dtype>*>&bottom,
      const vector<Blob<Dtype>*>& top);
 
virtual void Backward_cpu(const vector<Blob<Dtype>*>&top,
      const vector<bool>&propagate_down, const vector<Blob<Dtype>*>& bottom);
virtual void Backward_gpu(const vector<Blob<Dtype>*>&top,
      const vector<bool>&propagate_down, const vector<Blob<Dtype>*>& bottom);
 
Blob<Dtype> diff_ap_;  //cached for backward pass: f(x_i^a) - f(x_i^p)
Blob<Dtype> diff_an_;  //cached for backward pass: f(x_i^a) - f(x_i^n)
Blob<Dtype> diff_pn_;  //cached for backward pass: f(x_i^n) - f(x_i^p)

 
Blob<Dtype> diff_sq_ap_;  //cached for backward pass: ||f(x_i^a)-f(x_i^p)||^2
Blob<Dtype> diff_sq_an_;  //tmp storage for gpu forward pass: f(x_i^a)-f(x_i^p).^2 
 
Blob<Dtype> dist_sq_ap_;  //cached for backward pass:||f(x_i^a)-f(x_i^p)||^2
Blob<Dtype> dist_sq_an_;  //cached for backward pass:||f(x_i^a)-f(x_i^n)||^2
 
Blob<Dtype> summer_vec_;  //tmp storage for gpu forward pass[ 1, 1, 1,...,1 ]
//Blob<Dtype> dist_binary_; // tmp storage for gpu forward pass


Dtype margin;//距离参数a

};


}