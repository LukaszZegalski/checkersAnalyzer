import checkersAnalyzer as ca

if __name__ == '__main__':
    analyzer = ca.checkersAnalyzer(False)
    analyzer.read('./Picture_Lukasz/test.jpg')
    analyzer.detectBoard()
    analyzer.detectCircle()

