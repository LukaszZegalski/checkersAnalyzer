import checkersAnalyzer as ca

if __name__ == '__main__':
    analyzer = ca.checkersAnalyzer(False)
    analyzer.read('./Picture/22.jpg')
    analyzer.checkboardTransposition()
    analyzer.detectAreaBoardDistribution()
    analyzer.intoDictionary()
    analyzer.detectCircle()
    analyzer.drawBoard()