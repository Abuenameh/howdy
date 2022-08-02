#ifndef MODELS_H_
#define MODELS_H_

#include <vector>

#include <opencv2/videoio.hpp>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;

class face_detection_model
{

public:
    virtual ~face_detection_model() = default;

    std::vector<rectangle> operator()(cv::Mat &image, const int upsample_num_times);

    virtual std::vector<rectangle> detect(matrix<rgb_pixel> &image);
};

class cnn_face_detection_model_v1 : public face_detection_model
{

public:
    cnn_face_detection_model_v1(const std::string &model_filename);

    virtual ~cnn_face_detection_model_v1() = default;

    virtual std::vector<rectangle> detect(matrix<rgb_pixel> &image);

private:
    template <long num_filters, typename SUBNET>
    using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
    template <long num_filters, typename SUBNET>
    using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

    template <typename SUBNET>
    using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
    template <typename SUBNET>
    using rcon5 = relu<affine<con5<45, SUBNET>>>;

    using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

    net_type net;
};

class frontal_face_detector_model : public face_detection_model
{

public:
    frontal_face_detector_model();

    virtual ~frontal_face_detector_model() = default;

    virtual std::vector<rectangle> detect(matrix<rgb_pixel> &image);

private:
    frontal_face_detector detector;
};

class face_recognition_model_v1
{

public:
    face_recognition_model_v1(const std::string &model_filename);

    matrix<double, 0, 1> compute_face_descriptor(
        cv::Mat &image,
        const full_object_detection &face,
        const int num_jitters,
        float padding = 0.25);

    matrix<double, 0, 1> compute_face_descriptor(
        matrix<rgb_pixel> img,
        const full_object_detection &face,
        const int num_jitters,
        float padding = 0.25);

    matrix<double, 0, 1> compute_face_descriptor_from_aligned_image(
        matrix<rgb_pixel> img,
        const int num_jitters);

    std::vector<matrix<double, 0, 1>> compute_face_descriptors(
        matrix<rgb_pixel> img,
        const std::vector<full_object_detection> &faces,
        const int num_jitters,
        float padding = 0.25);

    std::vector<std::vector<matrix<double, 0, 1>>> batch_compute_face_descriptors(
        const std::vector<matrix<rgb_pixel>> &batch_imgs,
        const std::vector<std::vector<full_object_detection>> &batch_faces,
        const int num_jitters,
        float padding = 0.25);

    std::vector<matrix<double, 0, 1>> batch_compute_face_descriptors_from_aligned_images(
        const std::vector<matrix<rgb_pixel>> &batch_imgs,
        const int num_jitters);

private:
    dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> jitter_image(
        const matrix<rgb_pixel> &img,
        const int num_jitters);

    template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
    using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

    template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
    using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

    template <int N, typename SUBNET>
    using ares = relu<residual<block, N, affine, SUBNET>>;
    template <int N, typename SUBNET>
    using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

    template <typename SUBNET>
    using alevel0 = ares_down<256, SUBNET>;
    template <typename SUBNET>
    using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
    template <typename SUBNET>
    using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
    template <typename SUBNET>
    using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
    template <typename SUBNET>
    using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

    using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

    anet_type net;
};

class shape_predictor_model
{
public:
    shape_predictor_model(const std::string &model_filename);

    full_object_detection operator()(cv::Mat &image, const rectangle &box);

private:
    shape_predictor predictor;
};



#endif // MODELS_H_
