import checkersAnalyzer as ca

if __name__ == '__main__':
    analyzer = ca.checkersAnalyzer(True)
    analyzer.read('C:/Users/Piotr/Desktop/checkers.PNG')
    analyzer.threshold()
