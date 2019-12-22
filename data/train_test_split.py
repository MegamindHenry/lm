def save(name, lines):
    with open(name, 'w+', encoding='utf8') as fp:
        fp.write('\n\n'.join(lines))


tasa_doc = open('tasaDocs.txt', 'r', encoding='utf8').read().split('\n\n')

length = len(tasa_doc)
test_split = length - length//4

tasa_train = tasa_doc[:test_split]
tasa_test = tasa_doc[test_split:]

save('tasaTrain.txt', tasa_train)
save('tasaTest.txt', tasa_test)
