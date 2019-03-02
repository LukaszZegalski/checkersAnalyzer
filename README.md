Zasaday pracy: 
- Każdy pracuje na własnym branchu, 
- Branch łączony jest z masterem gdy dana funkcjonalność działa
- Piszemy testy

Kurs obsługi gita w skrócie: 
1. Tworzenie brancha i przełaczenie się na niego: git checkout -b nazwa brancha
2. Przełaczanie się pomiędzy branchami: git checkout nazwa brancha
3. Zaciągięcie zmian: git pull 
4. Wrzucenie brancha na remote (Czyli tak aby inni mogli go pobrać): git push origin nazwa brancha

Wrzucenie zmian (operacje powinny być wykonane dość szybko -tzn w 15/20 min): 
1. Wszystkie zmiany (te które chcesz) commitujesz na swoim branchu.
2. Przełaczasz się na master. Wpisujesz: git pull
3. Przełączasz się na swój branch: git checkout Twoja nazwa brancha
3. Wpisujesz: git rebase master
4. Przełaczasz się na master: git checkout master
5. Wpisujesz git merge <Twoja nazwa brancha>
6. Wpisujesz git pull  

