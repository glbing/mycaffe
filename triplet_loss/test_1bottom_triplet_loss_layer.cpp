#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/1bottom_triplet_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Triplet1LossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Triplet1LossLayerTest()
      : blob_bottom_data(new Blob<Dtype>(27,2,1,1)),
        blob_bottom_label(new Blob<Dtype>(27,1,1,1)),
        blob_top_loss(new Blob<Dtype>()) 
  {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_data);
    blob_bottom_vec_.push_back(blob_bottom_data);

    for (int i = 0; i < blob_bottom_label->count(); ++i) 
        blob_bottom_label->mutable_cpu_data()[i] = caffe_rng_rand() % 2;  // 0 or 1
    blob_bottom_vec_.push_back(blob_bottom_label);
    blob_top_vec_.push_back(blob_top_loss);
  }
  virtual ~Triplet1LossLayerTest() {
    delete blob_bottom_data;
    delete blob_top_loss;
  }
/*
  Blob<Dtype>* const blob_bottom_data_i_;
  Blob<Dtype>* const blob_bottom_data_j_;
  Blob<Dtype>* const blob_bottom_data_k_;
  Blob<Dtype>* const blob_bottom_y_;
*/
  
  Blob<Dtype>* const blob_bottom_data;
    Blob<Dtype>* const blob_bottom_label;
  Blob<Dtype>* const blob_top_loss;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(Triplet1LossLayerTest, TestDtypesAndDevices);

TYPED_TEST(Triplet1LossLayerTest, TestForward) 
{
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Triplet1LossLayer<Dtype> layer(layer_param);//triplet类
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);//前向计算
  // manually compute to compare
  const Dtype margin = layer_param.triplet_loss_param().margin();
  const Dtype num_triplets = layer_param.triplet_loss_param().num_triplets();
  const int num_set = this->blob_bottom_vec_[0]->num()/(2 + num_triplets); 
  const int dim = this->blob_bottom_vec_[0]->count()/this->blob_bottom_vec_[0]->num();//channels
  Dtype loss(0.0);//       
  //const Dtype *sampleW = this->blob_bottom_y_->cpu_data();   
                                                                                                                    
  for (int i = 0; i < num_set; ++i)
  {
    Dtype dist_sq_ap(0);
    Dtype dist_sq_an(0);  
    for(int j=0;j<dim;++j)
    {
      Dtype diff_ap = this->blob_bottom_data->cpu_data()[(int)((2+num_triplets)*i*dim+j)] -
          this->blob_bottom_data->cpu_data()[(int)(((2+num_triplets)*i+1)*dim+j)];
      Dtype diff_an = this->blob_bottom_data->cpu_data()[(int)((2+num_triplets)*i*dim+j)] -
          this->blob_bottom_data->cpu_data()[(int)(((2+num_triplets)*i+2)*dim+j)];
      dist_sq_ap+=diff_ap*diff_ap;
      dist_sq_an+=diff_an*diff_an;
    }
    loss += /*sampleW[i]**/std::max(Dtype(0.0), margin+dist_sq_ap-dist_sq_an);
  }
  loss = loss / static_cast<Dtype>(num_set) / Dtype(2);//
  EXPECT_NEAR(this->blob_top_loss->cpu_data()[0], loss, 1e-6);

}

TYPED_TEST(Triplet1LossLayerTest, TestGradient) {
 typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Triplet1LossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the first two bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  /*checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);*/
  
}
}  // namespace caffe
