#include <sys/syslog.h>
#include <syslog.h>

#include "utils.hh"
#include "models.hh"

std::vector<rectangle> face_detection_model::operator()(cv::Mat &image, const int upsample_num_times)
{
    pyramid_down<2> pyr;
    std::vector<rectangle> rects;

    matrix<rgb_pixel> dimage;
    convert_image(image, dimage);

    // Upsampling the image will allow us to detect smaller faces but will cause the
    // program to use more RAM and run longer.
    unsigned int levels = upsample_num_times;
    while (levels > 0)
    {
        levels--;
        pyramid_up(dimage, pyr);
    }

    auto dets = detect(dimage);

    // Scale the detection locations back to the original image size
    // if the image was upscaled.
    for (auto &&rect : dets)
    {
        rect = pyr.rect_down(rect, upsample_num_times);
        rects.push_back(rect);
    }

    return rects;
}

std::vector<rectangle> face_detection_model::detect(matrix<rgb_pixel> &image)
{
    return std::vector<rectangle>();
}

cnn_face_detection_model_v1::cnn_face_detection_model_v1(const std::string &model_filename)
{
    deserialize(model_filename) >> net;
}

std::vector<rectangle> cnn_face_detection_model_v1::detect(matrix<rgb_pixel> &image)
{
    std::vector<mmod_rect> dets = net(image);
    std::vector<rectangle> rects;
    for (auto &&d : dets)
    {
        rects.push_back(d.rect);
    }
    return rects;
}

frontal_face_detector_model::frontal_face_detector_model()
{
    detector = get_frontal_face_detector();
}

std::vector<rectangle> frontal_face_detector_model::detect(matrix<rgb_pixel> &image)
{
    return detector(image);
}

face_recognition_model_v1::face_recognition_model_v1(const std::string &model_filename)
{
    deserialize(model_filename) >> net;
}

matrix<double, 0, 1> face_recognition_model_v1::compute_face_descriptor(
    cv::Mat &image,
    const full_object_detection &face,
    const int num_jitters,
    float padding)
{
    matrix<rgb_pixel> img;
    convert_image(image, img);

    std::vector<full_object_detection> faces(1, face);
    return compute_face_descriptors(img, faces, num_jitters, padding)[0];
}

matrix<double, 0, 1> face_recognition_model_v1::compute_face_descriptor(
    matrix<rgb_pixel> img,
    const full_object_detection &face,
    const int num_jitters,
    float padding)
{
    std::vector<full_object_detection> faces(1, face);
    return compute_face_descriptors(img, faces, num_jitters, padding)[0];
}

matrix<double, 0, 1> face_recognition_model_v1::compute_face_descriptor_from_aligned_image(
    matrix<rgb_pixel> img,
    const int num_jitters)
{
    std::vector<matrix<rgb_pixel>> images{img};
    return batch_compute_face_descriptors_from_aligned_images(images, num_jitters)[0];
}

std::vector<matrix<double, 0, 1>> face_recognition_model_v1::compute_face_descriptors(
    matrix<rgb_pixel> img,
    const std::vector<full_object_detection> &faces,
    const int num_jitters,
    float padding)
{
    std::vector<matrix<rgb_pixel>> batch_img(1, img);
    std::vector<std::vector<full_object_detection>> batch_faces(1, faces);
    return batch_compute_face_descriptors(batch_img, batch_faces, num_jitters, padding)[0];
}

std::vector<std::vector<matrix<double, 0, 1>>> face_recognition_model_v1::batch_compute_face_descriptors(
    const std::vector<matrix<rgb_pixel>> &batch_imgs,
    const std::vector<std::vector<full_object_detection>> &batch_faces,
    const int num_jitters,
    float padding)
{

    if (batch_imgs.size() != batch_faces.size())
    {
        syslog(LOG_ERR, "The array of images and the array of array of locations must be of the same size");
        exit(1);
    }

    int total_chips = 0;
    for (const auto &faces : batch_faces)
    {
        total_chips += faces.size();
        for (const auto &f : faces)
        {
            if (f.num_parts() != 68 && f.num_parts() != 5)
            {
                syslog(LOG_ERR, "The full_object_detection must use the iBUG 300W 68 point face landmark style or dlib's 5 point style.");
                exit(1);
            }
        }
    }

    dlib::array<matrix<rgb_pixel>> face_chips;
    for (unsigned int i = 0; i < batch_imgs.size(); ++i)
    {
        auto &faces = batch_faces[i];
        auto &img = batch_imgs[i];

        std::vector<chip_details> dets;
        for (const auto &f : faces)
            dets.push_back(get_face_chip_details(f, 150, padding));
        dlib::array<matrix<rgb_pixel>> this_img_face_chips;
        extract_image_chips(img, dets, this_img_face_chips);

        for (auto &chip : this_img_face_chips)
            face_chips.push_back(chip);
    }

    std::vector<std::vector<matrix<double, 0, 1>>> face_descriptors(batch_imgs.size());
    if (num_jitters <= 1)
    {
        // extract descriptors and convert from float vectors to double vectors
        auto descriptors = net(face_chips, 16);
        auto next = std::begin(descriptors);
        for (unsigned int i = 0; i < batch_faces.size(); ++i)
        {
            for (unsigned int j = 0; j < batch_faces[i].size(); ++j)
            {
                face_descriptors[i].push_back(matrix_cast<double>(*next++));
            }
        }
    }
    else
    {
        // extract descriptors and convert from float vectors to double vectors
        auto fimg = std::begin(face_chips);
        for (unsigned int i = 0; i < batch_faces.size(); ++i)
        {
            for (unsigned int j = 0; j < batch_faces[i].size(); ++j)
            {
                auto &r = mean(mat(net(jitter_image(*fimg++, num_jitters), 16)));
                face_descriptors[i].push_back(matrix_cast<double>(r));
            }
        }
    }

    return face_descriptors;
}

std::vector<matrix<double, 0, 1>> face_recognition_model_v1::batch_compute_face_descriptors_from_aligned_images(
    const std::vector<matrix<rgb_pixel>> &batch_imgs,
    const int num_jitters)
{
    dlib::array<matrix<rgb_pixel>> face_chips;
    for (auto image : batch_imgs)
    {

        // Check for the size of the image
        if ((image.nr() != 150) || (image.nc() != 150))
        {
            syslog(LOG_ERR, "Unsupported image size, it should be of size 150x150. Also cropping must be done as `dlib.get_face_chip` would do it. \
                That is, centered and scaled essentially the same way.");
            exit(1);
        }

        face_chips.push_back(image);
    }

    std::vector<matrix<double, 0, 1>> face_descriptors;
    if (num_jitters <= 1)
    {
        // extract descriptors and convert from float vectors to double vectors
        auto descriptors = net(face_chips, 16);

        for (auto &des : descriptors)
        {
            face_descriptors.push_back(matrix_cast<double>(des));
        }
    }
    else
    {
        // extract descriptors and convert from float vectors to double vectors
        for (auto &fimg : face_chips)
        {
            auto &r = mean(mat(net(jitter_image(fimg, num_jitters), 16)));
            face_descriptors.push_back(matrix_cast<double>(r));
        }
    }
    return face_descriptors;
}

std::vector<matrix<rgb_pixel>> face_recognition_model_v1::jitter_image(
    const matrix<rgb_pixel> &img,
    const int num_jitters)
{
    std::vector<matrix<rgb_pixel>> crops;
    for (int i = 0; i < num_jitters; ++i)
        crops.push_back(dlib::jitter_image(img, rnd));
    return crops;
}

shape_predictor_model::shape_predictor_model(const std::string &model_filename)
{
    deserialize(model_filename) >> predictor;
}

full_object_detection shape_predictor_model::operator()(cv::Mat &image, const rectangle &box)
{
    matrix<rgb_pixel> img;
    convert_image(image, img);

    return predictor(img, box);
}
