#include <filesystem>
#include <regex>

#include <gtkmm.h>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "../../howdy/src/utils/argparse.hpp"
#include "../../howdy/src/process/process.hpp"

#define FMT_HEADER_ONLY
#include "../../howdy/src/fmt/core.h"

#include "../../howdy/src/utils.hh"
#include "../../howdy/src/string_utils/string_utils.hh"

namespace fs = std::filesystem;

using namespace TinyProcessLib;
using namespace std::literals;

class ModelColumns : public Gtk::TreeModelColumnRecord
{
public:
    ModelColumns()
    {
        add(name);
        add(recommended);
        add(path);
    }

    Gtk::TreeModelColumn<Glib::ustring> name;
    Gtk::TreeModelColumn<Glib::ustring> recommended;
    Gtk::TreeModelColumn<Glib::ustring> path;
};

class OnboardingWindow : public Gtk::Window
{
public:
    /*Initialize the sticky window*/
    OnboardingWindow()
    {
        // signal_delete_event().connect(sigc::mem_fun(this, &OnboardingWindow::exit));
        signal_delete_event().connect([this](GdkEventAny* event) { exit(); return true; });

        builder = Gtk::Builder::create();
        builder->add_from_file("/lib64/security/howdy-gtk/onboarding.glade");

        builder->get_widget<Gtk::Button>("scanbutton", scanbutton);
        scanbutton->signal_clicked().connect(sigc::mem_fun(this, &OnboardingWindow::on_scanbutton_click));
        builder->get_widget<Gtk::Button>("cancelbutton", cancelbutton);
        cancelbutton->signal_clicked().connect(sigc::mem_fun(this, &OnboardingWindow::exit));
        builder->get_widget<Gtk::Button>("finishbutton", finishbutton);
        finishbutton->signal_clicked().connect(sigc::mem_fun(this, &OnboardingWindow::exit));
        builder->get_widget<Gtk::Button>("nextbutton", nextbutton);
        nextbutton->signal_clicked().connect(sigc::mem_fun(this, &OnboardingWindow::go_next_slide));

        builder->get_widget<Gtk::Window>("onboardingwindow", window);

        Gtk::Box *slide;
        builder->get_widget<Gtk::Box>("slide0", slide);
        slides.push_back(slide);
        builder->get_widget<Gtk::Box>("slide1", slide);
        slides.push_back(slide);
        builder->get_widget<Gtk::Box>("slide2", slide);
        slides.push_back(slide);
        builder->get_widget<Gtk::Box>("slide3", slide);
        slides.push_back(slide);
        builder->get_widget<Gtk::Box>("slide4", slide);
        slides.push_back(slide);
        builder->get_widget<Gtk::Box>("slide5", slide);
        slides.push_back(slide);

        window->resize(500, 400);
        window->show_all();

        current_slide = 0;

        // Start GTK main loop
        Gtk::Main::run(*window);
    }

    void go_next_slide()
    {
        nextbutton->set_sensitive(false);

        slides[current_slide]->hide();
        slides[current_slide + 1]->show();
        current_slide += 1;

        if (current_slide == 1)
            execute_slide1();
        else if (current_slide == 2)
            Glib::signal_timeout().connect(sigc::mem_fun(this, &OnboardingWindow::execute_slide2), 10);
        else if (current_slide == 3)
            execute_slide3();
        else if (current_slide == 4)
            execute_slide4();
        else if (current_slide == 5)
            execute_slide5();
    }

    void execute_slide1()
    {
        builder->get_widget<Gtk::Label>("downloadoutputlabel", downloadoutputlabel);
        Gtk::EventBox *eventbox;
        builder->get_widget<Gtk::EventBox>("downloadeventbox", eventbox);
        eventbox->override_background_color(Gdk::RGBA("Black"), Gtk::STATE_FLAG_NORMAL);

        std::string lib_site("/lib64");
        if (!fs::exists(fs::status(lib_site + "/security/howdy/")))
        {
            downloadoutputlabel->set_text("Unable to find Howdy's installation location");
            return;
        }

        if (fs::exists(fs::status(lib_site + "/security/howdy/dlib-data/shape_predictor_5_face_landmarks.dat")))
        {
            downloadoutputlabel->set_text("Datafiles have already been downloaded!\nClick Next to continue");
            enable_next();
            return;
        }

        download_lines = std::deque<std::string>();

        downloadss = std::stringstream();
        proc = std::make_shared<Process>(
            "/bin/sh -c ./install.sh", lib_site + "/security/howdy/dlib-data", [&](const char *bytes, size_t n)
            { download_lines.push_back(std::string{bytes, n}); });

        read_download_line();
        Glib::signal_timeout().connect(sigc::mem_fun(this, &OnboardingWindow::read_download_line), 10);
    }

    bool read_download_line()
    {
        if (download_lines.size() > 10)
            download_lines.pop_front();

        std::string download_text = download_lines[0];
        for (int i = 1; i < download_lines.size(); i++)
            download_text += " "s + download_lines[i];
        downloadoutputlabel->set_text(download_text);

        int status;
        if (!proc->try_get_exit_status(status))
        {
            Glib::signal_timeout().connect(sigc::mem_fun(this, &OnboardingWindow::read_download_line), 10);
            return false;
        }

        // Wait for the process to finish and check the status code
        if (status != 0)
        {
            show_error("Error while downloading datafiles", download_text);
        }

        downloadoutputlabel->set_text("Done!\nClick Next to continue");
        enable_next();
        return false;
    }

    bool execute_slide2()
    {
        std::function<bool(cv::Mat &)> is_gray{[](cv::Mat &frame)
                                               {
                                                   for (auto iter = frame.begin<cv::Vec3f>(); iter != frame.end<cv::Vec3f>(); ++iter)
                                                   {
                                                       auto pixel = *iter;
                                                       if (!(pixel[0] == pixel[1] && pixel[1] == pixel[2]))
                                                           return false;
                                                   }
                                                   return true;
                                               }};

        std::vector<fs::path> device_ids;
        for (auto const &dir_entry : fs::directory_iterator{"/dev/v4l/by-path"})
        // for (auto const &dir_entry : fs::directory_iterator{"/home/abuenameh/v4l"})
            device_ids.push_back(dir_entry.path());
        std::vector<std::vector<std::string>> device_rows;

        if (device_ids.empty())
            show_error("No webcams found on system", "Please configure your camera yourself if you are sure a compatible camera is connected");

        // Loop though all devices
        for (auto &dev : device_ids)
        {
            std::this_thread::sleep_for(500ms);

            // The full path to the device is the default name
            std::string device_path = dev;
            std::string device_name = dev.filename();

            // Get the udevadm details to try to get a better name
            std::ostringstream oss;
            // Process udevadm_process("/bin/sh -c udevadm info -r --query=all -n " + device_path, "", [&](const char *bytes, size_t n)
            //                         { oss << std::string{bytes, n}; });
            Process udevadm_process("udevadm info -r --query=all -n " + device_path, "", [&](const char *bytes, size_t n)
                                    { oss << std::string{bytes, n}; });
            udevadm_process.get_exit_status();
            std::string udevadm = oss.str();

            // Loop though udevadm to search for a better name
            for (auto &line : split(udevadm, "\n"))
            {
                // Match it and encase it in quotes
                std::regex name_regex("product.*=(.*)$", std::regex::ECMAScript | std::regex::icase);
                std::smatch regex_result;
                if (std::regex_search(line, regex_result, name_regex))
                    device_name = regex_result[1];
            }

            cv::VideoCapture capture(device_path);
            cv::Mat frame;
            bool is_open = capture.read(frame);
            if (!is_open)
            {
                device_rows.push_back({device_name, device_path, "-9", "No, camera can't be opened"});
                continue;
            }

            if (!is_gray(frame))
            {
                device_rows.push_back({device_name, device_path, "-5", "No, not an infrared camera"});
                capture.release();
                continue;
            }

            device_rows.push_back({device_name, device_path, "5", "Yes, compatible infrared camera"});
            capture.release();
        }

        std::sort(device_rows.begin(), device_rows.end(), [](const std::vector<std::string> &a, const std::vector<std::string> &b)
                  { return std::stoi(a[2]) > std::stoi(b[2]); });

        Gtk::Label *loadinglabel;
        builder->get_widget<Gtk::Label>("loadinglabel", loadinglabel);
        Gtk::Box *devicelistbox;
        builder->get_widget<Gtk::Box>("devicelistbox", devicelistbox);

        treeview = Gtk::manage(new Gtk::TreeView());
        treeview->set_vexpand(true);

        // Set the coloums
        std::vector<std::string> columns{"Camera identifier or path", "Recommended"};
        for (int i = 0; i < columns.size(); i++)
        {
            auto cell = Gtk::make_managed<Gtk::CellRendererText>();
            cell->set_property("ellipsize", Pango::ELLIPSIZE_END);
            auto col = Gtk::make_managed<Gtk::TreeViewColumn>(columns[i], *cell);
            col->add_attribute(*cell, "text", i);
            treeview->append_column(*col);
        }

        // Add the treeview
        devicelistbox->add(*treeview);

        // Create a datamodel
        listmodel = Gtk::ListStore::create(model_columns);

        for (auto &device : device_rows)
        {
            auto iter = listmodel->append();
            auto row = *iter;
            row[model_columns.name] = device[0];
            row[model_columns.recommended] = device[3];
            row[model_columns.path] = device[1];
        }

        treeview->set_model(listmodel);
        treeview->set_cursor(Gtk::TreePath("0"));

        treeview->show();
        loadinglabel->hide();
        enable_next();

        return false;
    }

    void execute_slide3()
    {
        auto selection = treeview->get_selection();
        auto listmodel = selection->get_model();
        auto iter = selection->get_selected();

        if (!iter)
            show_error("Error selecting camera");

        auto row = *iter;
        std::string device_path = Glib::ustring(row[model_columns.path]);
        proc = std::make_shared<Process>(
            "howdy set device_path " + device_path, "");

        window->set_focus(*scanbutton);
    }

    void on_scanbutton_click()
    {
        int status = proc->get_exit_status();

        scan_dialog = std::make_shared<Gtk::MessageDialog>(*this, "", false, Gtk::MESSAGE_INFO, Gtk::BUTTONS_OK, true);
        scan_dialog->set_title("Creating Model");
        scan_dialog->set_message("Please look directly into the camera");
        scan_dialog->show_all();

        // Wait a bit to allow the user to read the dialog
        Glib::signal_timeout().connect(sigc::mem_fun(this, &OnboardingWindow::run_add), 600);
    }

    bool run_add()
    {
        std::ostringstream oss;
        Process howdy("howdy -y add", "", [&](const char *bytes, size_t n)
                      { oss << std::string{bytes, n}; });
        int status = howdy.get_exit_status();
        std::string output = oss.str();

        std::cout << "howdy add output:" << std::endl;
        std::cout << output << std::endl;

        scan_dialog->hide();

        if (status != 0)
            show_error("Can't save face model", output);

        Glib::signal_timeout().connect([this]()
                                       { go_next_slide(); return false; },
                                       10);

        return false;
    }

    void execute_slide4()
    {
        enable_next();
    }

    void execute_slide5()
    {
        Gtk::RadioButton *fast;
        builder->get_widget<Gtk::RadioButton>("radiofast", fast);
        Gtk::RadioButton *balanced;
        builder->get_widget<Gtk::RadioButton>("radiobalanced", balanced);
        Gtk::RadioButton *secure;
        builder->get_widget<Gtk::RadioButton>("radiosecure", secure);
        bool radio_selected = "";
        double radio_certanty = 5.0;

        if (fast->get_active())
            radio_certanty = 4.2;
        else if (balanced->get_active())
            radio_certanty = 3.5;
        else if (secure->get_active())
            radio_certanty = 2.2;
        else
            show_error("Error reading radio buttons");

        proc = std::make_shared<Process>(
            "howdy set certainty " + std::to_string(radio_certanty), "");

        nextbutton->hide();
        cancelbutton->hide();

        finishbutton->show();
        window->set_focus(*finishbutton);

        int status = proc->get_exit_status();

        if (status != 0)
            show_error("Error setting certainty", "Certainty is set to the default value, Howdy setup is complete");
    }

    void enable_next()
    {
        nextbutton->set_sensitive(true);
        window->set_focus(*nextbutton);
    }

    void show_error(std::string error, std::string secon = "")
    {
        Gtk::MessageDialog dialog(*this, "", false, Gtk::MESSAGE_ERROR, Gtk::BUTTONS_CLOSE, true);
        dialog.set_title("Howdy Error");
        dialog.set_message(error);
        dialog.set_secondary_text(secon);

        dialog.run();

        exit();
    }

    /*Cleanly exit*/
    void exit()
    {
        Gtk::Main::quit();
        ::exit(0);
    }

    Glib::RefPtr<Gtk::Builder> builder;
    Gtk::Window *window;
    Gtk::Button *nextbutton;
    Gtk::Button *cancelbutton;
    Gtk::Button *finishbutton;
    Gtk::Button *scanbutton;
    std::vector<Gtk::Box *> slides;
    int current_slide;
    Gtk::Label *downloadoutputlabel;
    std::deque<std::string> download_lines;
    std::stringstream downloadss;
    std::shared_ptr<Process> proc;
    Gtk::TreeView *treeview;
    ModelColumns model_columns;
    Glib::RefPtr<Gtk::ListStore> listmodel;
    std::shared_ptr<Gtk::MessageDialog> scan_dialog;
};

void onboarding_main(int argc, char *argv[])
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    Gtk::Main main(argc, argv);

    // Open the GTK window
    OnboardingWindow main_window;
}