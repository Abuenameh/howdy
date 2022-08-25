#include <filesystem>

#include <gtkmm.h>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <INIReader.h>

#include "../../howdy/src/utils/argparse.hpp"
#include "../../howdy/src/process/process.hpp"

#define FMT_HEADER_ONLY
#include "../../howdy/src/fmt/core.h"

#include "../../howdy/src/utils.hh"
#include "../../howdy/src/string_utils/string_utils.hh"

namespace fs = std::filesystem;

using namespace TinyProcessLib;
using namespace std::literals;

void onboarding_main(int argc, char *argv[]);

class ModelColumns : public Gtk::TreeModelColumnRecord
{
public:
    ModelColumns()
    {
        add(id);
        add(created);
        add(label);
    }

    Gtk::TreeModelColumn<Glib::ustring> id;
    Gtk::TreeModelColumn<Glib::ustring> created;
    Gtk::TreeModelColumn<Glib::ustring> label;
};

class MainWindow : public Gtk::Window
{
public:
    /*Initialize the sticky window*/
    MainWindow()
    {
        signal_delete_event().connect([this](GdkEventAny *event)
                                      { exit(); return true; });

        builder = Gtk::Builder::create();
        builder->add_from_file("/lib64/security/howdy-gtk/main.glade");

        builder->get_widget<Gtk::ComboBoxText>("userlist", userlist);
        userlist->signal_changed().connect(sigc::mem_fun(this, &MainWindow::on_user_change));

        Gtk::Label *label;
        builder->get_widget<Gtk::Label>("githublabel", label);
        label->signal_activate_link().connect(sigc::mem_fun(this, &MainWindow::on_about_link));
        builder->get_widget<Gtk::Label>("donatelabel", label);
        label->signal_activate_link().connect(sigc::mem_fun(this, &MainWindow::on_about_link));

        Gtk::Notebook *notebook;
        builder->get_widget<Gtk::Notebook>("notebook", notebook);
        notebook->signal_switch_page().connect(sigc::mem_fun(this, &MainWindow::on_page_switch));
        Gtk::Button *adduserbutton;
        builder->get_widget<Gtk::Button>("adduserbutton", adduserbutton);
        adduserbutton->signal_clicked().connect(sigc::mem_fun(this, &MainWindow::on_user_add));
        Gtk::Button *addbutton;
        builder->get_widget<Gtk::Button>("addbutton", addbutton);
        addbutton->signal_clicked().connect(sigc::mem_fun(this, &MainWindow::on_model_add));
        Gtk::Button *deletebutton;
        builder->get_widget<Gtk::Button>("deletebutton", deletebutton);
        deletebutton->signal_clicked().connect(sigc::mem_fun(this, &MainWindow::on_model_delete));

        Gtk::Window *window;
        Gtk::Box *modellistbox;
        builder->get_widget<Gtk::Window>("mainwindow", window);
        builder->get_widget<Gtk::Box>("modellistbox", modellistbox);
        builder->get_widget<Gtk::Image>("opencvimage", opencvimage);

        // Init capture for video tab
        capture = nullptr;

        // Create a treeview that will list the model data
        treeview = Gtk::manage(new Gtk::TreeView());
        treeview->set_vexpand(true);

        // Set the coloums
        std::vector<std::string> columns{"ID", "Created", "Label"};
        for (int i = 0; i < columns.size(); i++)
        {
            auto renderer = Gtk::make_managed<Gtk::CellRendererText>();
            auto col = Gtk::make_managed<Gtk::TreeViewColumn>(columns[i], *renderer);
            col->add_attribute(*renderer, "text", i);
            treeview->append_column(*col);
        }
        // treeview->append_column("ID", model_columns.id);
        // treeview->append_column("Created", model_columns.created);
        // treeview->append_column("Label", model_columns.label);

        // Add the treeview
        modellistbox->add(*treeview);

        std::vector<fs::path> filelist;
        for (auto const &dir_entry : fs::directory_iterator{"/lib64/security/howdy/models"})
            filelist.push_back(dir_entry.path());
        active_user = "";

        userlistitems = 0;

        for (auto &file : filelist)
        {
            userlist->append(std::string(file.stem()));
            userlistitems += 1;

            if (active_user == "")
                active_user = file.stem();
        }

        userlist->set_active(0);

        window->show_all();

        // Start GTK main loop
        Gtk::Main::run(*window);
    }

    /*(Re)load the model list*/
    void load_model_list()
    {

        // Execute the list commond to get the models
        std::ostringstream oss;
        Process howdy("howdy --plain -U '" + active_user + "' list", "", [&](const char *bytes, size_t n)
                      { oss << std::string{bytes, n}; });
        int status = howdy.get_exit_status();
        std::string output = oss.str();

        // Create a datamodel
        listmodel = Gtk::ListStore::create(model_columns);

        // If there was no error
        if (status == 0)
        {
            // Split the output per line
            std::vector<std::string> lines = split(output, "\n");

            // Add the models to the datamodel
            for (auto &line : lines)
            {
                std::vector<std::string> cols = split(line, ",");
                auto iter = listmodel->append();
                auto row = *iter;
                row[model_columns.id] = cols.size() > 0 ? cols[0] : "";
                row[model_columns.created] = cols.size() > 1 ? cols[1] : "";
                row[model_columns.label] = cols.size() > 2 ? cols[2] : "";
            }
        }

        treeview->set_model(listmodel);
    }

    /*Open links on about page as a non-root user*/
    bool on_about_link(const Glib::ustring &uri)
    {
        char *user_p = getlogin();
        if (!user_p)
        {
            user_p = std::getenv("SUDO_USER");
        }
        if (user_p)
        {
            std::string user = user_p;
            Process openuri("sudo -u " + user + " timeout 10 xdg-open " + uri, "");
        }
        return true;
    }

    /*Cleanly exit*/
    void exit()
    {
        if (capture)
            capture->release();
        Gtk::Main::quit();
        ::exit(0);
    }

#include "tab_models.incl"

#include "tab_video.incl"

    Glib::RefPtr<Gtk::Builder> builder;
    Gtk::ComboBoxText *userlist;
    int userlistitems;
    std::string active_user;
    Gtk::TreeView *treeview;
    ModelColumns model_columns;
    Glib::RefPtr<Gtk::ListStore> listmodel;
    std::shared_ptr<cv::VideoCapture> capture;
    Gtk::Image *opencvimage;
    double scaling_factor;
};

void elevate(int argc, char *argv[], bool graphical = true)
{
    if (getuid() == 0)
        return;

    std::vector<std::string> args{argv, argv + argc};
    std::vector<std::vector<std::string>> commands;

    if (graphical)
    {
        if (std::getenv("DISPLAY"))
        {
            // TODO Get full paths
            std::vector<std::string> pkcommand{"/usr/bin/pkexec"};
            pkcommand.insert(pkcommand.end(), args.begin(), args.end());
            commands.push_back(pkcommand);
            std::vector<std::string> gkcommand{"/usr/bin/gksudo"};
            gkcommand.insert(gkcommand.end(), args.begin(), args.end());
            commands.push_back(gkcommand);
            std::vector<std::string> kdecommand{"/usr/bin/kdesudo"};
            kdecommand.insert(kdecommand.end(), args.begin(), args.end());
            commands.push_back(kdecommand);
        }
    }

    std::vector<std::string> sudocommand{"/usr/bin/sudo"};
    sudocommand.insert(sudocommand.end(), args.begin(), args.end());
    commands.push_back(sudocommand);

    for (auto &arguments : commands)
    {
        std::vector<const char *> argv_ptrs;
        argv_ptrs.reserve(arguments.size() + 1);
        for (auto &arg : arguments)
            argv_ptrs.emplace_back(arg.c_str());
        argv_ptrs.emplace_back(nullptr);

        execv(arguments[0].c_str(), const_cast<char *const *>(argv_ptrs.data()));
        if (errno != ENOENT || arguments[0].ends_with("/sudo"))
        {
            throw std::runtime_error(std::strerror(errno));
        }
    }
}

void window_main(int argc, char *argv[], bool force_onboarding)
{
    // Make sure we quit on a SIGINT
    // signal.signal(signal.SIGINT, signal.SIG_DFL)

    // Make sure we run as sudo
    elevate(argc, argv);

    // If no models have been created yet or when it is forced, start the onboarding
    if (force_onboarding || !fs::exists(fs::status("/lib64/security/howdy/models")))
    {
        onboarding_main(argc, argv);

        exit(0);
    }

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    Gtk::Main main(argc, argv);

    // Open the GTK window
    MainWindow main_window;
}