#include <sys/syslog.h>
#include <syslog.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <map>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
// #include <dlib/image_processing.h>

#include <INIReader.h>

#include "compare.hh"

#include "json.hpp"

// #define FMT_HEADER_ONLY
// #include "fmt/core.h"

const std::string PATH = "/lib64/security/howdy";

using json = nlohmann::json;
using namespace dlib;
// using namespace cv;

namespace fs = std::filesystem;

typedef std::chrono::time_point<std::chrono::system_clock> time_point;

inline void exit_code(int code)
{
    /*Terminate gtk_proc*/
    exit(code);
}

inline time_point now()
{
    return std::chrono::system_clock::now();
}

inline void convert_image(cv::Mat &iimage, matrix<rgb_pixel> &oimage)
{
    // cv_image<bgr_pixel> cimage(image);
    // matrix<rgb_pixel> dimage;
    // assign_image(dimage, cimage);
    // if (is_image<unsigned char>(img))
    //     assign_image(image, numpy_image<unsigned char>(img));
    // else if (is_image<rgb_pixel>(img))
    //     assign_image(image, numpy_image<rgb_pixel>(img));
    // else
    // {
    //     syslog(LOG_ERR, "Unsupported image type, must be 8bit gray or RGB image.");
    //     exit_code(1);
    // }

    if (iimage.channels() == 1)
    {
        assign_image(oimage, cv_image<unsigned char>(iimage));
    }
    else if (iimage.channels() == 3)
    {
        assign_image(oimage, cv_image<bgr_pixel>(iimage));
    }
    else
    {
        syslog(LOG_ERR, "Unsupported image type, must be 8bit gray or RGB image.");
        exit_code(1);
    }
}

// inline Eigen::MatrixXd toEigenMatrix(std::vector<std::vector<double>> vectors){
// 		Eigen::MatrixXd M(vectors.size(), vectors.front().size());
// 	for(size_t i = 0; i < vectors.size(); i++)
// 		for(size_t j = 0; j < vectors.front().size(); j++)
// 			M(i,j) = vectors[i][j];
//             return M;
// }

class face_detection_model
{

public:
    virtual ~face_detection_model() = default;

    std::vector<rectangle> operator()(cv::Mat &image, const int upsample_num_times)
    {
        pyramid_down<2> pyr;
        std::vector<rectangle> rects;

        // cv_image<bgr_pixel> cimage(image);
        matrix<rgb_pixel> dimage;
        convert_image(image, dimage);
        // assign_image(dimage, cimage);

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

    virtual std::vector<rectangle> detect(matrix<rgb_pixel> &image)
    {
        return std::vector<rectangle>();
    }
};

class cnn_face_detection_model_v1 : public face_detection_model
{

public:
    cnn_face_detection_model_v1(const std::string &model_filename)
    {
        deserialize(model_filename) >> net;
    }

    virtual ~cnn_face_detection_model_v1() = default;

    virtual std::vector<rectangle> detect(matrix<rgb_pixel> &image)
    {
        std::vector<mmod_rect> dets = net(image);
        std::vector<rectangle> rects;
        for (auto &&d : dets)
        {
            rects.push_back(d.rect);
        }
        return rects;
    }

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
    frontal_face_detector_model()
    {
        detector = get_frontal_face_detector();
    }

    virtual ~frontal_face_detector_model() = default;

    virtual std::vector<rectangle> detect(matrix<rgb_pixel> &image)
    {
        return detector(image);
    }

private:
    frontal_face_detector detector;
};

class face_recognition_model_v1
{

public:
    face_recognition_model_v1(const std::string &model_filename)
    {
        deserialize(model_filename) >> net;
    }

    matrix<double, 0, 1> compute_face_descriptor(
        cv::Mat &image,
        const full_object_detection &face,
        const int num_jitters,
        float padding = 0.25)
    {
        // cv_image<bgr_pixel> cimage(image);
        matrix<rgb_pixel> img;
        convert_image(image, img);
        // assign_image(img, cimage);

        std::vector<full_object_detection> faces(1, face);
        return compute_face_descriptors(img, faces, num_jitters, padding)[0];
    }

    matrix<double, 0, 1> compute_face_descriptor(
        matrix<rgb_pixel> img,
        const full_object_detection &face,
        const int num_jitters,
        float padding = 0.25)
    {
        std::vector<full_object_detection> faces(1, face);
        return compute_face_descriptors(img, faces, num_jitters, padding)[0];
    }

    matrix<double, 0, 1> compute_face_descriptor_from_aligned_image(
        matrix<rgb_pixel> img,
        const int num_jitters)
    {
        std::vector<matrix<rgb_pixel>> images{img};
        return batch_compute_face_descriptors_from_aligned_images(images, num_jitters)[0];
    }

    std::vector<matrix<double, 0, 1>> compute_face_descriptors(
        matrix<rgb_pixel> img,
        const std::vector<full_object_detection> &faces,
        const int num_jitters,
        float padding = 0.25)
    {
        std::vector<matrix<rgb_pixel>> batch_img(1, img);
        std::vector<std::vector<full_object_detection>> batch_faces(1, faces);
        return batch_compute_face_descriptors(batch_img, batch_faces, num_jitters, padding)[0];
    }

    std::vector<std::vector<matrix<double, 0, 1>>> batch_compute_face_descriptors(
        const std::vector<matrix<rgb_pixel>> &batch_imgs,
        const std::vector<std::vector<full_object_detection>> &batch_faces,
        const int num_jitters,
        float padding = 0.25)
    {

        if (batch_imgs.size() != batch_faces.size())
        {
            syslog(LOG_ERR, "The array of images and the array of array of locations must be of the same size");
            exit_code(1);
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
                    exit_code(1);
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

    std::vector<matrix<double, 0, 1>> batch_compute_face_descriptors_from_aligned_images(
        const std::vector<matrix<rgb_pixel>> &batch_imgs,
        const int num_jitters)
    {
        dlib::array<matrix<rgb_pixel>> face_chips;
        for (auto image : batch_imgs)
        {

            // matrix<rgb_pixel> image;
            // assign_image(image, img);
            // if (is_image<unsigned char>(img))
            //     assign_image(image, numpy_image<unsigned char>(img));
            // else if (is_image<rgb_pixel>(img))
            //     assign_image(image, numpy_image<rgb_pixel>(img));
            // else
            // {
            //     syslog(LOG_ERR, "Unsupported image type, must be 8bit gray or RGB image.");
            //     exit_code(1);
            // }

            // Check for the size of the image
            if ((image.nr() != 150) || (image.nc() != 150))
            {
                syslog(LOG_ERR, "Unsupported image size, it should be of size 150x150. Also cropping must be done as `dlib.get_face_chip` would do it. \
                That is, centered and scaled essentially the same way.");
                exit_code(1);
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

private:
    dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> jitter_image(
        const matrix<rgb_pixel> &img,
        const int num_jitters)
    {
        std::vector<matrix<rgb_pixel>> crops;
        for (int i = 0; i < num_jitters; ++i)
            crops.push_back(dlib::jitter_image(img, rnd));
        return crops;
    }

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
    shape_predictor_model(const std::string &model_filename)
    {
        deserialize(model_filename) >> predictor;
    }

    full_object_detection operator()(cv::Mat &image, const rectangle &box)
    {
        cv_image<bgr_pixel> cimage(image);
        matrix<rgb_pixel> img;
        assign_image(img, cimage);

        return predictor(img, box);
    }

private:
    shape_predictor predictor;
};

class VideoCapture
{

public:
    /*
    Creates a new VideoCapture instance depending on the settings in the
    provided config file.
    */
    VideoCapture(INIReader config_) : config(config_)
    {
        if (!fs::exists(fs::status(config.Get("video", "device_path", ""))))
        {
            if (config.GetBoolean("video", "warn_no_device", true))
            {
                syslog(LOG_ERR, "Howdy could not find a camera device at the path specified in the config file.");
                exit_code(1);
            }
        }

        // Create reader
        // The internal video recorder
        // Start video capture on the IR camera through OpenCV
        internal = cv::VideoCapture(config.Get("video", "device_path", ""), cv::CAP_V4L);

        // Force MJPEG decoding if true
        if (config.GetBoolean("video", "force_mjpeg", false))
        {
            // Set a magic number, will enable MJPEG but is badly documentated
            internal.set(cv::CAP_PROP_FOURCC, 1196444237);
        }

        // Set the frame width and height if requested
        // The frame width
        fw = config.GetInteger("video", "frame_width", -1);
        // The frame height
        fh = config.GetInteger("video", "frame_height", -1);
        if (fw != -1)
            internal.set(cv::CAP_PROP_FRAME_WIDTH, fw);
        if (fh != -1)
            internal.set(cv::CAP_PROP_FRAME_HEIGHT, fh);

        // Request a frame to wake the camera up
        internal.grab();
    }

    /*
    Frees resources when destroyed
    */
    ~VideoCapture()
    {
        internal.release();
    }

    /*
    Release cameras
    */
    void release()
    {
        internal.release();
    }

    /*
    Reads a frame, returns the frame and an attempted grayscale conversion of
    the frame in a tuple:

    (frame, grayscale_frame)

    If the grayscale conversion fails, both items in the tuple are identical.
    */
    void read_frame(cv::Mat &frame, cv::Mat &gsframe)
    {
        bool ret = internal.read(frame);
        if (!ret)
        {
            syslog(LOG_ERR, "Failed to read camera specified in the 'device_path' config option, aborting");
            exit_code(1);
        }

        // Convert from color to grayscale
        cv::cvtColor(frame, gsframe, cv::COLOR_BGR2GRAY);
    }

    double get(int propId)
    {
        return internal.get(propId);
    }

    bool set(int propId, double value)
    {
        return internal.set(propId, value);
    }

    int fw;
    int fh;

private:
    INIReader config;
    cv::VideoCapture internal;
};

template <typename T>
struct op_vector_to_matrix
{
    /*!
        This object defines a matrix expression that holds a reference to a std::vector<T>
        and makes it look like a column vector.  Thus it enables you to use a std::vector
        as if it were a dlib::matrix.

    !*/
    op_vector_to_matrix(const std::vector<T> &vect_) : vect(vect_) {}

    const std::vector<T> &vect;

    // This expression wraps direct memory accesses so we use the lowest possible cost.
    const static long cost = 1;

    const static long NR = 0; // We don't know the length of the vector until runtime.  So we put 0 here.
    const static long NC = 1; // We do know that it only has one column (since it's a vector)
    typedef T type;
    // Since the std::vector doesn't use a dlib memory manager we list the default one here.
    typedef default_memory_manager mem_manager_type;
    // The layout type also doesn't really matter in this case.  So we list row_major_layout
    // since it is a good default.
    typedef row_major_layout layout_type;

    // Note that we define const_ret_type to be a reference type.  This way we can
    // return the contents of the std::vector by reference.
    typedef const T &const_ret_type;
    const_ret_type apply(long r, long) const { return vect[r]; }

    long nr() const { return vect.size(); }
    long nc() const { return 1; }

    // This expression never aliases anything since it doesn't contain any matrix expression (it
    // contains only a std::vector which doesn't count since you can't assign a matrix expression
    // to a std::vector object).
    template <typename U>
    bool aliases(const matrix_exp<U> &) const { return false; }
    template <typename U>
    bool destructively_aliases(const matrix_exp<U> &) const { return false; }
};

template <typename T>
const matrix_op<op_vector_to_matrix<T>> vector_to_matrix(
    const std::vector<T> &vector)
{
    typedef op_vector_to_matrix<T> op;
    return matrix_op<op>(op(vector));
}

int main(int argc, char *argv[])
{
    std::map<std::string, time_point> start_times;
    std::map<std::string, std::chrono::duration<double>> timings;

    start_times["st"] = now();

    openlog("howdy-auth", 0, LOG_AUTHPRIV);

    // Make sure we were given an username to test against
    if (argc < 2)
    {
        exit_code(12);
    }

    // The username of the user being authenticated
    char *user = argv[1];
    // The model file contents
    // models = [];
    // Encoded face models
    // encodings = [];
    std::vector<matrix<double, 0, 1>> encodings;
    // Amount of ignored 100% black frames
    int black_tries = 0;
    // Amount of ingnored dark frames
    int dark_tries = 0;
    // Total amount of frames captured
    int frames = 0;
    // Captured frames for snapshot capture
    // snapframes = [];
    // Tracks the lowest certainty value in the loop
    int lowest_certainty = 10;
    // Face recognition/detection instances
    // face_detector = None;
    // pose_predictor = None;
    // face_encoder = None;

    // Try to load the face model from the models folder
    if (!fs::exists(fs::status(PATH + "/models/" + user + ".dat")))
    {
        syslog(LOG_ERR, "Model file not found for user %s", user);
        exit_code(10);
    }
    std::ifstream f(PATH + "/models/" + user + ".dat");
    json models = json::parse(f);
    for (auto &model : models)
    {
        for (auto &row : model["data"])
        {
            encodings.push_back(vector_to_matrix(std::vector<double>(row)));
        }
    }

    // Check if the file contains a model
    if (models.size() < 1)
    {
        exit_code(10);
    }

    // Read config from disk
    INIReader config(PATH + "/config.ini");

    // Error out if we could not read the config file
    if (config.ParseError() != 0)
    {
        syslog(LOG_ERR, "Failed to parse the configuration file: %d", config.ParseError());
        exit_code(10);
    }

    // Get all config values needed
    bool use_cnn = config.GetBoolean("core", "use_cnn", false);
    int timeout = config.GetInteger("video", "timeout", 5);
    double dark_threshold = config.GetReal("video", "dark_threshold", 50.0);
    double video_certainty = config.GetReal("video", "certainty", 3.5) / 10;
    bool end_report = config.GetBoolean("debug", "end_report", false);
    bool capture_failed = config.GetBoolean("snapshots", "capture_failed", false);
    bool capture_successful = config.GetBoolean("snapshots", "capture_successful", false);
    bool gtk_stdout = config.GetBoolean("debug", "gtk_stdout", false);
    int rotate = config.GetInteger("video", "rotate", 0);

    // Send the gtk outupt to the terminal if enabled in the config
    // gtk_pipe = sys.stdout if gtk_stdout else subprocess.DEVNULL

    // Start the auth ui, register it to be always be closed on exit
    // try:
    //     gtk_proc = subprocess.Popen(["../howdy-gtk/src/init.py", "--start-auth-ui"], stdin=subprocess.PIPE, stdout=gtk_pipe, stderr=gtk_pipe)
    //     atexit.register(exit)
    // except FileNotFoundError:
    //     pass

    // Write to the stdin to redraw ui
    // send_to_ui("M", _("Starting up..."))

    // Save the time needed to start the script
    timings["in"] = now() - start_times["st"];

    // Import face recognition, takes some time
    start_times["ll"] = now();

    if (!fs::is_regular_file(fs::status(PATH + "/dlib-data/shape_predictor_5_face_landmarks.dat")))
    {
        syslog(LOG_ERR, "Data files have not been downloaded");
        exit_code(1);
    }

    face_detection_model face_detector;
    if (use_cnn)
    {
        face_detector = cnn_face_detection_model_v1(PATH + "/dlib-data/mmod_human_face_detector.dat");
    }
    else
    {
        face_detector = frontal_face_detector_model();
    }

    // Start the others regardless
    shape_predictor_model pose_predictor = shape_predictor_model(PATH + "/dlib-data/shape_predictor_5_face_landmarks.dat");
    face_recognition_model_v1 face_encoder = face_recognition_model_v1(PATH + "/dlib-data/dlib_face_recognition_resnet_model_v1.dat");

    // Note the time it took to initialize detectors
    timings["ll"] = now() - start_times["ll"];

    // Start video capture on the IR camera
    start_times["ic"] = now();

    VideoCapture video_capture(config);

    // Read exposure from config to use in the main loop
    int exposure = config.GetInteger("video", "exposure", -1);

    // Note the time it took to open the camera
    timings["ic"] = now() - start_times["ic"];

    // Fetch the max frame height
    double max_height = config.GetReal("video", "max_height", 0.0);

    // Get the height of the image (which would be the width if screen is portrait oriented)
    double height;
    height = video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (rotate == 2)
    {
        height = video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
    }
    if (height == 0)
    {
        height = 1;
    }
    // Calculate the amount the image has to shrink
    double scaling_factor = max_height / height;
    if (scaling_factor == 0)
    {
        scaling_factor = 1;
    }

    // Initiate histogram equalization
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));

    // Let the ui know that we're ready
    // send_to_ui("M", _("Identifying you..."))

    // Start the read loop
    frames = 0;
    int valid_frames = 0;
    start_times["fr"] = now();
    double dark_running_total = 0;

    while (true)
    {
        // Increment the frame count every loop
        frames += 1;

        // Form a string to let the user know we're real busy
        std::string ui_subtext = "Scanned " + std::to_string(valid_frames - dark_tries) + " frames";
        if (dark_tries > 1)
        {
            ui_subtext += " (skipped " + std::to_string(dark_tries) + " dark frames)";
        }
        // Show it in the ui as subtext
        // send_to_ui("S", ui_subtext)

        // Stop if we've exceded the time limit
        if (std::chrono::duration<double>(now() - start_times["fr"]).count() > timeout)
        {
            // Create a timeout snapshot if enabled
            if (capture_failed)
            {
                // make_snapshot(_("FAILED"))
            }

            if (dark_tries == valid_frames)
            {
                syslog(LOG_ERR, "All frames were too dark, please check dark_threshold in config");
                syslog(LOG_ERR, "Average darkness: %f, Threshold: %f", dark_running_total / std::max(1, valid_frames), dark_threshold);
                exit_code(13);
            }
            else
            {
                exit_code(11);
            }
        }

        // Grab a single frame of video
        cv::Mat tempframe;
        cv::Mat frame, gsframe;
        video_capture.read_frame(frame, tempframe);
        clahe->apply(tempframe, gsframe);

        // If snapshots have been turned on
        if (capture_failed || capture_successful)
        {
            // Start capturing frames for the snapshot
            // if len(snapframes) < 3:
            // 	snapframes.append(frame)
        }

        // Create a histogram of the image with 8 values
        cv::Mat hist;
        cv::calcHist(std::vector<cv::Mat>{gsframe}, std::vector<int>{0}, cv::Mat(), hist, std::vector<int>{8}, std::vector<float>{0, 256});
        // All values combined for percentage calculation
        double hist_total = cv::sum(hist)[0];

        // Calculate frame darkness
        double darkness = (hist.at<float>(0) / hist_total * 100);

        // If the image is fully black due to a bad camera read,
        // skip to the next frame
        if ((hist_total == 0) or (darkness == 100))
        {
            black_tries += 1;
            continue;
        }

        dark_running_total += darkness;
        valid_frames += 1;
        // If the image exceeds darkness threshold due to subject distance,
        // skip to the next frame
        if (darkness > dark_threshold)
        {
            dark_tries += 1;
            continue;
        }

        // If the height is too high
        if (scaling_factor != 1)
        {
            // Apply that factor to the frame
            cv::resize(frame, tempframe, cv::Size(), scaling_factor, scaling_factor, cv::INTER_AREA);
            frame = tempframe;
            cv::resize(gsframe, tempframe, cv::Size(), scaling_factor, scaling_factor, cv::INTER_AREA);
            gsframe = tempframe;
        }
        // If camera is configured to rotate = 1, check portrait in addition to landscape
        if (rotate == 1)
        {
            if (frames % 3 == 1)
            {
                cv::rotate(frame, tempframe, cv::ROTATE_90_COUNTERCLOCKWISE);
                frame = tempframe;
                cv::rotate(gsframe, tempframe, cv::ROTATE_90_COUNTERCLOCKWISE);
                gsframe = tempframe;
            }
            if (frames % 3 == 2)
            {
                cv::rotate(frame, tempframe, cv::ROTATE_90_CLOCKWISE);
                frame = tempframe;
                cv::rotate(gsframe, tempframe, cv::ROTATE_90_CLOCKWISE);
                gsframe = tempframe;
            }
        }
        // If camera is configured to rotate = 2, check portrait orientation
        else if (rotate == 2)
        {
            if (frames % 2 == 0)
            {
                cv::rotate(frame, tempframe, cv::ROTATE_90_COUNTERCLOCKWISE);
                frame = tempframe;
                cv::rotate(gsframe, tempframe, cv::ROTATE_90_COUNTERCLOCKWISE);
                gsframe = tempframe;
            }
            else
            {
                cv::rotate(frame, tempframe, cv::ROTATE_90_CLOCKWISE);
                frame = tempframe;
                cv::rotate(gsframe, tempframe, cv::ROTATE_90_CLOCKWISE);
                gsframe = tempframe;
            }
        }

        // Get all faces from that frame as encodings
        // Upsamples 1 time
        std::vector<rectangle> face_locations = face_detector(gsframe, 1);

        // Loop through each face
        for (auto &&fl : face_locations)
        {
            // Fetch the faces in the image
            auto face_landmark = pose_predictor(frame, fl);
            auto face_encoding = face_encoder.compute_face_descriptor(frame, face_landmark, 1);

            // Match this found face against a known face
            std::vector<double> matches;
            std::transform(encodings.begin(), encodings.end(), std::back_inserter(matches), [face_encoding](auto &encoding)
                           { return sqrt(sum(squared(encoding - face_encoding))); });

            // Get best match
            int match_index = static_cast<int>(std::distance(matches.begin(), min_element(matches.begin(), matches.end())));
            double match = matches[match_index];

            // Update certainty if we have a new low
            if (lowest_certainty > match)
            {
                lowest_certainty = match;
            }

            // Check if a match that's confident enough
            if (0 < match && match < video_certainty)
            {
                timings["tt"] = now() - start_times["st"];
                timings["fl"] = now() - start_times["fr"];

                // If set to true in the config, print debug text
                if (end_report)
                {
                    /*
                    Helper function to print a timing from the list
                    */
                    auto syslog_timing = [&timings](std::string label, std::string k)
                    {
                        syslog(LOG_INFO, "  %s: %dms", label.c_str(), int(round(timings[k].count() * 1000)));
                    };

                    // Print a nice timing report
                    syslog(LOG_INFO, "Time spent");
                    syslog_timing("Starting up", "in");
                    syslog(LOG_INFO, "  Open cam + load libs: %dms", int(round(std::max(timings["ll"].count(), timings["ic"].count()) * 1000)));
                    syslog_timing("  Opening the camera", "ic");
                    syslog_timing("  Importing recognition libs", "ll");
                    syslog_timing("Searching for known face", "fl");
                    syslog_timing("Total time", "tt");

                    syslog(LOG_INFO, "\nResolution");
                    double width = video_capture.fw;
                    if (width == 0)
                    {
                        width = 1;
                    }
                    syslog(LOG_INFO, "  Native: %dx%d", int(height), int(width));
                    // Save the new size for diagnostics
                    // scale_height, scale_width = frame.shape[:2];
                    // syslog(LOG_INFO, "  Used: %dx%d", scale_height, scale_width);

                    // Show the total number of frames and calculate the FPS by deviding it by the total scan time
                    syslog(LOG_INFO, "\nFrames searched: %d (%.2f fps)", frames, frames / timings["fl"].count());
                    syslog(LOG_INFO, "Black frames ignored: %d ", black_tries);
                    syslog(LOG_INFO, "Dark frames ignored: %d ", dark_tries);
                    syslog(LOG_INFO, "Certainty of winning frame: %.3f", match * 10);

                    syslog(LOG_INFO, "Winning model: %d (\"%s\")", match_index, std::string(models[match_index]["label"]).c_str());
                }
                // Make snapshot if enabled
                if (capture_successful)
                {
                    // make_snapshot(_("SUCCESSFUL"));
                }

                // Run rubberstamps if enabled
                if (config.GetBoolean("rubberstamps", "enabled", false))
                {
                    // import rubberstamps

                    // send_to_ui("S", "")

                    // if "gtk_proc" not in vars() : 
                    // gtk_proc = None

                    //  rubberstamps.execute(config, gtk_proc, {
                    //  "video_capture" : video_capture,
                    //  "face_detector" : face_detector,
                    //  "pose_predictor" : pose_predictor,
                    //  "clahe" : clahe
                    //  })
                }

                // End peacefully
                exit_code(0);
            }
        }
        if (exposure != -1)
        {
            // For a strange reason on some cameras (e.g. Lenoxo X1E) setting manual exposure works only after a couple frames
            // are captured and even after a delay it does not always work. Setting exposure at every frame is reliable though.
            video_capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 1.0); // 1 = Manual
            video_capture.set(cv::CAP_PROP_EXPOSURE, double(exposure));
        }
    }
}