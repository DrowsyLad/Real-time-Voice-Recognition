cr_sal = 7
m_sal = 9
r_sal = 10
cr_ex = 3
m_ex = 2
r_ex = 1
salary = []
expenses = []
for a in range(3):
    sal = int(input("salary: "))
    salary.append(sal)
for b in range(3):
    ex = int(input("expenses: "))
    expenses.append(ex)


length = len(salary)

def find_highest_savings(salary, expenses):
    for i in range(length):
        salary[i] = salary[i] - expenses[i]
    print(salary)
    salary.sort()
    k=2
    print(salary[k-1])   

find_highest_savings(salary, expenses)