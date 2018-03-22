import argparse


# sub-command functions
def create_data(args):
    print("downloading to: {}".format(args.data_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="topic model evaluation")
    subparsers = parser.add_subparsers(title='available sub-commands',
                                       help="sub-command help",
                                       dest="subparser")

    # "create" subparser
    parser_create = subparsers.add_parser('create',
                                          description='download and pre-process test data for evaluation')
    parser_create.add_argument('--data-dir', type=str, dest="data_dir",
                               help="destination directory for corpus data")
    parser_create.set_defaults(func=create_data)

    args = parser.parse_args()
    if args.subparser is None:
        parser.print_help()
    else:
        args.func(args)

