#include "../../howdy/src/utils/argparse.hpp"

void auth_main(int argc, char *argv[]);

void window_main(int argc, char *argv[], bool force_onboarding);

int main(int argc, char *argv[])
{
    setenv("DISPLAY", ":0", 0);
    setenv("NO_AT_BRIDGE", "1", 1);

    argparse::ArgumentParser parser(argc, argv);
    parser.add_argument("--start-auth-ui")
        .action("store_true");
    parser.add_argument("--force-onboarding")
        .action("store_true");

    argparse::Namespace args = parser.parse_args();

    if (args.get<bool>("--start-auth-ui"))
    {
        auth_main(argc, argv);
    }
    else
    {
        window_main(argc, argv, args.get<bool>("--force-onboarding"));
    }
}