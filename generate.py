import argparse
from utils.generator import ArithmeticGenerator, MixedArithmeticGenerator

parser = argparse.ArgumentParser(description='This script generates arithmetic problems.')
parser.add_argument('--output-dir', type=str, default='output',
                    help='Directory to save outputs')
parser.add_argument('--max-bound', type=int, default=100,
                    help='upper bound of operands (default: 100)')
parser.add_argument('--ops', nargs='+', type=str, default=['+', '-', '*'],
                    help='List of operators, default: (+, -, *)')
parser.add_argument('--test-size', type=float, default=0.3,
                    help='Proportion of test set, default: 0.3')
parser.add_argument('--seed', type=int, default=1234,
                    help='Set random seed, default: 1234')
parser.add_argument('--mixed', action='store_true',
                    help='generate problems with 2 or more operators')


def main(args):
    args = parser.parse_args(args)
    Generator = MixedArithmeticGenerator if args.mixed else ArithmeticGenerator
    gen = Generator(output_root=args.output_dir, bound=args.max_bound,
                    ops=args.ops, seed=args.seed,
                    test_size=args.test_size)
    gen.generate()
    gen.save()


if __name__ == '__main__':
    args = None
    args = ['--output-dir', r'./tmp', '--max-bound', '50', '--seed', '4111',
            '--test-size', '0.2', '--mixed']
    main(args)
