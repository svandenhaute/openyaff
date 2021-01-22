import argparse


def test():
    print('works!')


def main():
    parser = argparse.ArgumentParser(
            description='conversion and testing of YAFF force fields to OpenMM-compatible format',
            )
    parser.add_argument(
            'mode',
            action='store',
            help='determines mode of operation: init, compare, convert, test',
            )
    args = parser.parse_args()


    if args.mode == 'test':
        test()
    return 0
