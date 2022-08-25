#include <grp.h>

#include <filesystem>
#include <regex>

#include <gtkmm.h>
#include <cairo/cairo.h>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "../../howdy/src/utils/argparse.hpp"
#include "../../howdy/src/process/process.hpp"

#define FMT_HEADER_ONLY
#include "../../howdy/src/fmt/core.h"

#include "../../howdy/src/utils.hh"
#include "../../howdy/src/utils/string.hpp"

namespace fs = std::filesystem;

using namespace TinyProcessLib;
using namespace std::literals;

// Set window size constants
int windowWidth = 400;
int windowHeight = 100;

class StickyWindow : public Gtk::Window
{
public:
    /*Initialize the sticky window*/
    StickyWindow() : message("Loading...  ")
    {
        // Get the absolute or relative path to the logo file
        std::string logo_path = "/lib64/security/howdy-gtk/logo.png";
        if (!fs::exists(fs::status(logo_path)))
            logo_path = "./logo.png";

        // Create image and calculate scale size based on image size
        logo_surface = Cairo::ImageSurface::create_from_png(logo_path);
        logo_ratio = double(windowHeight - 20) / double(logo_surface->get_height());

        // Set the title of the window
        set_title("Howdy Authentication");

        // Set a bunch of options to make the window stick and be on top of everything
        stick();
        set_gravity(Gdk::GRAVITY_STATIC);
        set_resizable(false);
        set_keep_above(true);
        set_app_paintable(true);
        set_skip_pager_hint(true);
        set_skip_taskbar_hint(true);
        set_can_focus(false);
        set_can_default(false);
        unset_focus();
        set_type_hint(Gdk::WINDOW_TYPE_HINT_NOTIFICATION);
        set_decorated(0);

        // Listen for a window redraw
        signal_draw().connect(sigc::mem_fun(this, &StickyWindow::draw));
        // Listen for a force close or click event and exit
        signal_delete_event().connect([this](GdkEventAny *event)
                                      { exit(); return true; });
        signal_button_press_event().connect([this](GdkEventButton *event)
                                            { exit(); return true; });
        signal_button_release_event().connect([this](GdkEventButton *event)
                                              { exit(); return true; });

        // Create a GDK drawing, restricts the window size
        Gtk::DrawingArea *darea = Gtk::make_managed<Gtk::DrawingArea>();
        darea->set_size_request(windowWidth, windowHeight);
        add(*darea);

        // Get the default screen
        Glib::RefPtr<Gdk::Screen> screen = Gdk::Screen::get_default();
        Glib::RefPtr<Gdk::Visual> visual = screen->get_rgba_visual();
        gtk_widget_set_visual(GTK_WIDGET(gobj()), visual->gobj());

        // Move the window to the center top of the default window, where a webcam usually is
        move((screen->get_width() / 2) - (windowWidth / 2), 0);

        // Show window and force a resize again
        show_all();
        resize(windowWidth, windowHeight);

        // Add a timeout to catch input passed from compare.py
        Glib::signal_timeout().connect(sigc::mem_fun(this, &StickyWindow::catch_stdin), 100);

        // Start GTK main loop
        Gtk::Main::run(*this);
    }

    /*Draw the UI*/
    bool draw(const Cairo::RefPtr<Cairo::Context> &ctx)
    {
        // Change cursor to the kill icon
        get_window()->set_cursor(Gdk::Cursor::create(Gdk::CursorType::PIRATE));

        // Draw a semi transparent background
        ctx->set_source_rgba(0, 0, 0, .7);
        ctx->set_operator(Cairo::OPERATOR_SOURCE);
        ctx->paint();
        ctx->set_operator(Cairo::OPERATOR_OVER);

        // Position and draw the logo
        ctx->translate(15, 10);
        ctx->scale(logo_ratio, logo_ratio);
        ctx->set_source(logo_surface, 0, 0);
        ctx->paint();

        // Calculate main message positioning, as the text is heigher if there's a subtext
        if (subtext != "")
            ctx->move_to(380, 145);
        else
            ctx->move_to(380, 175);

        // Draw the main message
        ctx->set_source_rgba(255, 255, 255, .9);
        ctx->set_font_size(80);
        ctx->select_font_face("Arial", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_NORMAL);
        ctx->show_text(message);

        // Draw the subtext if there is one
        if (subtext != "")
        {
            ctx->move_to(380, 210);
            ctx->set_source_rgba(230, 230, 230, .8);
            ctx->set_font_size(40);
            ctx->select_font_face("Arial", Cairo::FONT_SLANT_NORMAL, Cairo::FONT_WEIGHT_NORMAL);
            ctx->show_text(subtext);
        }

        return true;
    }

    /*Catch input from stdin and redraw*/
    bool catch_stdin()
    {
        // Wait for a line on stdin
        std::string comm;
        std::getline(std::cin, comm);

        // If the line is not empty
        if (comm != "")
        {
            // Parse a message
            if (comm[0] == 'M')
                message = trim(comm.substr(2));
            // Parse subtext
            if (comm[0] == 'S')
                subtext = trim(comm.substr(2));
        }

        // Redraw the ui
        queue_draw();

        // Fire this function again in 10ms, as we're waiting on IO in readline anyway
        Glib::signal_timeout().connect(sigc::mem_fun(this, &StickyWindow::catch_stdin), 10);

        return false;
    }

    /*Cleanly exit*/
    void exit()
    {
        Gtk::Main::quit();
    }

    Cairo::RefPtr<Cairo::ImageSurface> logo_surface;
    double logo_ratio;
    std::string message;
    std::string subtext;
};

void deelevate()
{
    gid_t newgid = getgid(), oldgid = getegid();
    uid_t newuid = getuid(), olduid = geteuid();

    /* If root privileges are to be dropped, be sure to pare down the ancillary
     * groups for the process before doing anything else because the setgroups(  )
     * system call requires root privileges.  Drop ancillary groups regardless of
     * whether privileges are being dropped temporarily or permanently.
     */
    if (!olduid)
        setgroups(1, &newgid);

    if (newgid != oldgid)
    {
        if (setregid(newgid, newgid) == -1)
            abort();
    }

    if (newuid != olduid)
    {
        if (setreuid(newuid, newuid) == -1)
            abort();
    }

    /* verify that the changes were successful */
    if (newgid != oldgid && (setegid(oldgid) != -1 || getegid() != newgid))
        abort();
    if (newuid != olduid && (seteuid(olduid) != -1 || geteuid() != newuid))
        abort();
}

void auth_main(int argc, char *argv[])
{
    deelevate();

    Gtk::Main main(argc, argv);

    // Open the GTK window
    StickyWindow window;
}