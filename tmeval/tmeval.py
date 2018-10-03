import argparse
import corpora.generate.uci as cgu


# sub-command functions
def create_data(opts: argparse.Namespace):
    """
    Create/download corpora

    :param opts: argparse.Namespace
    :return:
    """
    
    corpora_to_download = opts.corpora
    download_all = opts.all
    data_dir = opts.data_dir

    if len(corpora_to_download) and download_all:
        raise ValueError("either specify corpora to download or --all, not both")

    if download_all:
        corpora_to_download = ['uci/kos', 'uci/nytimes']

    print("downloading following {} corpus: {}".format(len(corpora_to_download),
                                                       corpora_to_download))
    print("downloading to: {}".format(data_dir))

    for corpus in corpora_to_download:
        print("downloading {}".format(corpus))
        source_name, corpus_name = corpus.split("/")
        if source_name == 'uci':
            cgu.generate_mmcorpus_files(corpus_name, data_dir)
        else:
            raise ValueError("unknown source for corpus: {}".format(corpus))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="topic model evaluation")
    subparsers = parser.add_subparsers(title='available sub-commands',
                                       help="sub-command help",
                                       dest="subparser")

    # "create" subparser
    parser_create = subparsers.add_parser('create',
                                          description='download and pre-process test data for evaluation')
    parser_create.add_argument('corpora', type=str, nargs='*',
                               help="corpora to download")

    parser_create.add_argument('--all', default=False, action='store_true',
                               help="download all known corpora")

    parser_create.add_argument('--data-dir', type=str, dest="data_dir",
                               help="destination directory for corpus data")

    parser_create.set_defaults(func=create_data)

    args = parser.parse_args()
    if args.subparser is None:
        parser.print_help()
    else:
        args.func(args)

