#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {
template <typename Dtype>
  class Triplet1LossLayer : public LossLayer<Dtype> {
   public:
    explicit Triplet1LossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual inline int ExactNumBottomBlobs() const { return 2; }//
    virtual inline const char* type() const { return "Triplet1Loss"; }
    /**
     * Unlike most loss layers, in the Triplet1LossLayer we can backpropagate
     * to the first three inputs.
     */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return bottom_index != 1;
    }

   protected:
    /// @copydoc Triplet1LossLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    /*
     \begin{eqnarray}
     \mathcal{L}_{tri}(s_i,s_j,s_k) = max(0,1-\frac{||f(x_i)-f(x_k)||_2^2}{||f(x_i)-f(x_j)||_2^2+m})
     \end{eqnarray}®
     where $ f(x) $ is the input of the loss layer for sample $ x $ and m is the margin for triplet.
     Denote that $D_{ij}=||f(x_i)-f(x_j)||_2^2$ and $D_{ik}=||f(x_i)-f(x_k)||_2^2$,
     so the partial differential equations for the input of triplet loss layer are:
     \begin{eqnarray}
     \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_i)}=
     &\frac{D_{ik}(f(x_i)-f(x_j))-(D_{ij}+m)(f(x_i)-f(x_k))}{(D_{ij}+m)^2} \nonumber \\
     \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_j)}=
     &\frac{D_{ik}(f(x_j)-f(x_i))}{(D_{ij}+m)^2} \nonumber \\
     \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_k)}=
     &\frac{f(x_i)-f(x_k)}{D_{ij}+m}
     \end{eqnarray}®
     */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    Blob<Dtype> diff_ap_;  // cached for backward pass
    Blob<Dtype> diff_an_;
    Blob<Dtype> diff_pn_;

    Blob<Dtype> diff_sq_ap_;
    Blob<Dtype> diff_sq_an_;

    Blob<Dtype> dist_sq_ap_;
    Blob<Dtype> dist_sq_an_;


    //Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
    
    Dtype margin;//距离参数a
    int num_triplets;//应该negtive的个数
  };

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_