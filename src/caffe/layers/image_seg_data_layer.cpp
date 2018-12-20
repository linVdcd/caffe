#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_seg_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

    template <typename Dtype>
    ImageSegDataLayer<Dtype>::~ImageSegDataLayer<Dtype>() {
      this->StopInternalThread();
    }

    template <typename Dtype>
    void ImageSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
      const int new_height = this->layer_param_.image_seg_data_param().new_height();
      const int new_width  = this->layer_param_.image_seg_data_param().new_width();
      const bool is_color  = this->layer_param_.image_seg_data_param().is_color();
//      const int label_type = this->layer_param_.image_seg_data_param().label_type();
      string root_folder = this->layer_param_.image_seg_data_param().root_folder();

      TransformationParameter transform_param = this->layer_param_.transform_param();
      CHECK(transform_param.has_mean_file() == false) <<
                                                      "ImageSegDataLayer does not support mean file";
      CHECK((new_height == 0 && new_width == 0) ||
            (new_height > 0 && new_width > 0)) << "Current implementation requires "
              "new_height and new_width to be set at the same time.";

      // Read the file with filenames and labels
      const string& source = this->layer_param_.image_seg_data_param().source();
      LOG(INFO) << "Opening file " << source;
      std::ifstream infile(source.c_str());
      string line;
      string im_fn,mask_fn;
      size_t pos1,pos2;
      int label;
      while (std::getline(infile, line)) {
        pos2 = line.find_last_of(' ');
        pos1 = line.find_first_of(' ');
        CHECK(pos1!=pos2) <<"Data list must: img.jpg mask.jpg label";
        im_fn = line.substr(0,pos1);
        mask_fn = line.substr(pos1+1,pos2-pos1-1);
        label = atoi(line.substr(pos2 + 1).c_str());
        lines_.push_back(std::make_pair(im_fn, std::make_pair(mask_fn,label)));
      }

      if (this->layer_param_.image_seg_data_param().shuffle()) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        const unsigned int prefetch_rng_seed = caffe_rng_rand();
        prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
        ShuffleImages();
      }
      LOG(INFO) << "A total of " << lines_.size() << " images.";

      lines_id_ = 0;
      // Check if we would need to randomly skip a few data points
      if (this->layer_param_.image_seg_data_param().rand_skip()) {
        unsigned int skip = caffe_rng_rand() %
                            this->layer_param_.image_seg_data_param().rand_skip();
        LOG(INFO) << "Skipping first " << skip << " data points.";
        CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
        lines_id_ = skip;
      }
      // Read an image, and use it to initialize the top blob.
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                        new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

      const int channels = cv_img.channels();
      const int height = cv_img.rows;
      const int width = cv_img.cols;
      // image
      //const int crop_size = this->layer_param_.transform_param().crop_size();
      int crop_width = 0;
      int crop_height = 0;
      CHECK(transform_param.has_crop_size() )
      << "Must either specify crop_size or both crop_height and crop_width.";
      if (transform_param.has_crop_size()) {
        crop_width = transform_param.crop_size();
        crop_height = transform_param.crop_size();
      }

      const int batch_size = this->layer_param_.image_seg_data_param().batch_size();
      if (crop_width > 0 && crop_height > 0) {
        top[0]->Reshape(batch_size, channels, crop_height, crop_width);
        this->transformed_data_.Reshape(batch_size, channels, crop_height, crop_width);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
          this->prefetch_[i]->data_.Reshape(batch_size, channels, crop_height, crop_width);
        }

        //mask label
        top[1]->Reshape(batch_size, 1, crop_height, crop_width);
        this->transformed_seg_.Reshape(batch_size, 1, crop_height, crop_width);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
          this->prefetch_[i]->seg_.Reshape(batch_size, 1, crop_height, crop_width);
        }
      } else {
        top[0]->Reshape(batch_size, channels, height, width);
        this->transformed_data_.Reshape(batch_size, channels, height, width);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
          this->prefetch_[i]->data_.Reshape(batch_size, channels, height, width);
        }

        //label
        top[1]->Reshape(batch_size, 1, height, width);
        this->transformed_seg_.Reshape(batch_size, 1, height, width);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
          this->prefetch_[i]->seg_.Reshape(batch_size, 1, height, width);
        }
      }
      // image dimensions, for each image, stores (img_height, img_width)
        vector<int> label_shape(1, batch_size);
      top[2]->Reshape(label_shape);
      for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(label_shape);
      }

      LOG(INFO) << "output data size: " << top[0]->num() << ","
                << top[0]->channels() << "," << top[0]->height() << ","
                << top[0]->width();
      // mask label
      LOG(INFO) << "output label size: " << top[1]->num() << ","
                << top[1]->channels() << "," << top[1]->height() << ","
                << top[1]->width();

    }

    template <typename Dtype>
    void ImageSegDataLayer<Dtype>::ShuffleImages() {
      caffe::rng_t* prefetch_rng =
              static_cast<caffe::rng_t*>(prefetch_rng_->generator());
      shuffle(lines_.begin(), lines_.end(), prefetch_rng);
    }

// This function is called on prefetch thread
    template <typename Dtype>
    void ImageSegDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
      CPUTimer batch_timer;
      batch_timer.Start();
      double read_time = 0;
      double trans_time = 0;
      CPUTimer timer;
      CHECK(batch->data_.count());
      CHECK(this->transformed_data_.count());

      Dtype* top_data     = batch->data_.mutable_cpu_data();
      Dtype* top_seg    = batch->seg_.mutable_cpu_data();
      Dtype* top_label = batch->label_.mutable_cpu_data();

      const int max_height = batch->data_.height();
      const int max_width  = batch->data_.width();

      ImageSegDataParameter image_seg_data_param = this->layer_param_.image_seg_data_param();
      const int batch_size = image_seg_data_param.batch_size();
      const int new_height = image_seg_data_param.new_height();
      const int new_width  = image_seg_data_param.new_width();
      //const int label_type = this->layer_param_.image_seg_data_param().label_type();
//      const int ignore_label = image_seg_data_param.ignore_label();
      const bool is_color  = image_seg_data_param.is_color();
      string root_folder   = image_seg_data_param.root_folder();

      const int lines_size = lines_.size();
      for (int item_id = 0; item_id < batch_size; ++item_id) {
        std::vector<cv::Mat> cv_img_seg;

        // get a blob
        timer.Start();
        CHECK_GT(lines_size, lines_id_);

        int img_row, img_col;
        cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                              new_height, new_width, is_color));

        if (!cv_img_seg[0].data) {
          DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].first;
        }

        cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].second.first,
                                            new_height, new_width, false));
      if (!cv_img_seg[1].data) {
        DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id_].second.first;
      }

        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset;
        offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);

        offset = batch->seg_.offset(item_id);
        this->transformed_seg_.set_cpu_data(top_seg + offset);

        this->data_transformer_->TransformImgAndSeg(cv_img_seg,
                                                    &(this->transformed_data_), &(this->transformed_seg_),
                                                    -1);
        top_label[item_id] = lines_[lines_id_].second.second;

        trans_time += timer.MicroSeconds();

        // go to the next std::vector<int>::iterator iter;
        lines_id_++;
        if (lines_id_ >= lines_size) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          lines_id_ = 0;
          if (this->layer_param_.image_seg_data_param().shuffle()) {
            ShuffleImages();
          }
        }
      }
      batch_timer.Stop();
      DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
      DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
      DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    INSTANTIATE_CLASS(ImageSegDataLayer);
    REGISTER_LAYER_CLASS(ImageSegData);

}  // namespace caffe
#endif  // USE_OPENCV