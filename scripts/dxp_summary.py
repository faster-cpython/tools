# 

from collections import Counter


def main():
    filename = None
    stats = Counter()
    counts = Counter()
    with open("dxpstats.txt") as f:
        for line in f:
            words = line.split()
            match words:
                case []:
                    pass
                case ["Processing", filename]:
                    pass  # Save 'filename'
                case [prevop, "-->", nextop, _, percent]:
                    key = prevop, nextop
                    fraction = float(percent.rstrip("%")) / 100
                    stats[key] += fraction
                    counts[key] += 1
                case _:
                    print("What is", repr(line))
    table = [(stats[key] / counts[key], key) for key in stats]
    table.sort(reverse=True)
    total = 0
    for avg, (prevop, nextop) in table[:20]:
        total += avg
        print(f"{prevop:<20s} --> {nextop:<20s} {100*avg:6.2f}%")
    print(f"Total: {100*total:.2f}% (something weird going on here)")


if __name__ == "__main__":
    main()
