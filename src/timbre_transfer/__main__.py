import click

from timbre_transfer.train import train

@click.group()
def main():
    pass


main.command()(train)


if __name__ == "__main__":
    main()