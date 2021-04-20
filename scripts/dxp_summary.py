# 

from collections import Counter


def main():
    filename = None
    stats = Counter()
    count = 0
    with open("dxpstats.txt") as f:
        for line in f:
            words = line.split()
            match words:
                case []:
                    pass
                case ["Processing", filename]:
                    count += 1  # Save 'filename'
                case [lastop, "-->", nextop, _, percent]:
                    key = lastop, nextop
                    fraction = float(percent.rstrip("%")) / 100
                    stats[key] += fraction
                case _:
                    print("What is", repr(line))
    table = [(stats[key] / count, key) for key in stats]
    table.sort(reverse=True)
    total = 0
    for avg, (lastop, nextop) in table[:20]:
        total += avg
        print(f"{lastop:<20s} --> {nextop:<20s} {100*avg:6.2f}%")
    print(f"Total: {100*total:.2f}%")


if __name__ == "__main__":
    main()
