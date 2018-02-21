import argparse


# sub-command functions
def data_download(args):
    print("downloading to: {}".format(args.data_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_data = subparsers.add_parser('data')
    data_subparsers = parser_data.add_subparsers()
    parser_data_download = data_subparsers.add_parser('download')
    parser_data_download.add_argument('--data-dir', type=str, dest="data_dir")
    parser_data_download.set_defaults(func=data_download)

    args = parser.parse_args()
    args.func(args)

