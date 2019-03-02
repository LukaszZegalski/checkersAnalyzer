import checkersAnalyzer as ca

if __name__ == '__main__':
    analyzer = ca.checkersAnalyzer(True)
    analyzer.read('./Picture/chess.jpg')
    analyzer.threshlod()

