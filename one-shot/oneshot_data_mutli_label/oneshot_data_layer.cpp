#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/oneshot_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
OneshotDataLayer<Dtype>::~OneshotDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void OneshotDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int n_way = this->layer_param_.oneshot_data_param().n_way();
  const int k_shot = this->layer_param_.oneshot_data_param().k_shot();
  const int batch_size = this->layer_param_.oneshot_data_param().batch_size();

  CHECK_EQ(n_way*k_shot*2, batch_size)<<"the size of (set_b + set_b) must be equal to batch_size";

  const int new_height = this->layer_param_.oneshot_data_param().new_height();
  const int new_width  = this->layer_param_.oneshot_data_param().new_width();
  const bool is_color  = this->layer_param_.oneshot_data_param().is_color();
  string root_folder = this->layer_param_.oneshot_data_param().root_folder();

  const int label_size = this->layer_param_.oneshot_data_param().label_size();


  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.oneshot_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  //##############################################################
  std::string filename;
  while (std::getline(infile, line)) {//stream infile的每行读入到 string line
    std::istringstream str_list(line);//以空格将line分割
    str_list>>filename;
    //std::cout<<filename<<std::endl;
    int label;
    std::vector<int> labels;  
    while(str_list>>label)
    {
      //std::cout<<label<<std::endl;
      labels.push_back(label);//
    }
    //std::cout<<labels.size()<<std::endl;
    lines_.push_back(std::make_pair(filename,labels));//构造lines
  }

  CHECK(!lines_.empty()) << "File is empty";
  CHECK_EQ(lines_.size()%batch_size, 0)<<"the size of samples must be divisible by batch_size";
  //lines_  的元素为 txt的每一行的 path + label
  //由lines_生成 sets_
  
  //std::vector<std::string, int> temp;
  vector<std::pair<std::string, std::vector<int> > >::iterator iter_1;
  vector<std::pair<std::string, std::vector<int> > >::iterator iter_2;
  for(int i=0; i< lines_.size()/batch_size; ++i)
  {
    iter_1=lines_.begin()+i*batch_size;
    iter_2=lines_.begin()+(i+1)*batch_size;
    std::vector<std::pair<std::string, std::vector<int> > > temp(iter_1,iter_2);
    sets_.push_back(temp);
  }

  if (this->layer_param_.oneshot_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }

  LOG(INFO) << "A total of " << sets_.size()*batch_size << " images.";

  //从sets_恢复lines_
  for(int i=0;i<sets_.size();++i)
  {
    for(int j=0;j<batch_size;++j)
    {
      lines_[i*batch_size+j] = sets_[i][j];
    }
  }
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.oneshot_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.oneshot_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);

  // Reshape prefetch_data and top[0] according to the batch_size.
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  //vector<int> label_shape(1, batch_size);
  //################################################
  vector<int> label_shape(2,batch_size);
  label_shape[1]=label_size;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void OneshotDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(sets_.begin(), sets_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void OneshotDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  OneshotDataParameter oneshot_data_param = this->layer_param_.oneshot_data_param();
  const int batch_size = oneshot_data_param.batch_size();
  const int new_height = oneshot_data_param.new_height();
  const int new_width = oneshot_data_param.new_width();
  const bool is_color = oneshot_data_param.is_color();
  string root_folder = oneshot_data_param.root_folder();

  //###################
  const int label_size = this->layer_param_.oneshot_data_param().label_size();


  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    //##################################################
    //prefetch_label[item_id] = lines_[lines_id_].second;
    //std::cout<<lines_[lines_id_].second.size()<<std::endl;
    CHECK_EQ(label_size,lines_[lines_id_].second.size())<<"label_size not matching the proto setting";
    for (int i=0;i<label_size;i++)
      prefetch_label[item_id*label_size+i] = lines_[lines_id_].second[i];

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.oneshot_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(OneshotDataLayer);
REGISTER_LAYER_CLASS(OneshotData);

}  // namespace caffe
#endif  // USE_OPENCV
