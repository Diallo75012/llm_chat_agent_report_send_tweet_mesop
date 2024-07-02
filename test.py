a = ['b', 'c', 'e', 'f', 'd']
print("a: ", a)


for elem in a[:]:
  print(elem)
  a.remove(elem)

print("a: ", a)
