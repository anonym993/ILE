from collections import defaultdict


def one_time_reasoning(entity, rules, new_tri, in_map, out_map):
    for e1 in entity:
        for rule_index, (rule, p) in enumerate(rules):
            if len(rule) == 3:
                if rule[0].startswith('-'):
                    e2_list = in_map[(rule[0][1:], e1)]
                else:
                    e2_list = out_map[(e1, rule[0])]

                if len(e2_list) > 0:
                    for e2 in e2_list:
                        if rule[1].startswith('-'):
                            e3_list = in_map[(rule[1][1:], e2)]
                        else:
                            e3_list = out_map[(e2, rule[1])]
                        for e3 in e3_list:
                            new_tri.add((e1, rule[2], e3))
                else:
                    continue


            elif len(rule) == 2:

                if rule[0].startswith('-'):
                    e2_list = in_map[(rule[0][1:], e1)]
                else:
                    e2_list = out_map[(e1, rule[0])]

                for e2 in e2_list:
                    new_tri.add((e1, rule[1], e2))




def dense_your_graph(input, rules):
    new_triples = set()
    out_map = defaultdict(list)
    in_map = defaultdict(list)
    entity = set()

    with open(input, 'r', encoding='utf8') as f:
        for line in f:
            e1, r, e2 = line.strip().split("\t")
            out_map[(e1, r)].append(e2)
            in_map[(r, e2)].append(e1)
            entity.add(e1)
            entity.add(e2)

    entity = list(entity)

    for i in entity:
        one_time_reasoning(i, rules, new_triples, in_map, out_map)

    return new_triples


if __name__ == '__main__':


    input_tri = './datasets/DB100K/_train.txt'

    train_set = set()

    with open(input_tri) as f_train:
        for line in f_train:
            h, r, t = line.strip().split('\t')
            train_set.add((h, r, t))


    out_new_triples = './datasets/DB100K/new_triple.txt'
    rules_path = './datasets/DB100K/Rules.txt'

    rules = []

    with open(rules_path, 'r', encoding='utf8') as f:
        for line in f:
            rule, p = line.strip().split('\t')
            rules.append(tuple([tuple(rule.split(',')), float(p)]))

    f_out = open(out_new_triples, 'w')

    for i in range(len(rules)):

        newline = ''
        rules2 = rules[i:i + 1]
        _, p = rules2[0]
        newline += str(p) + '\t'
        new_triples = dense_your_graph(input_tri, rules2)
        new_triples = new_triples - train_set


        for h, r, t in new_triples:
            newline += h + '@$' + r + '@$' + t + '\t'
        newline = newline[:-1] + '\n'

        f_out.write(newline)
