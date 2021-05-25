

def rule_transform(rule):
    rule_body, rule_head = rule.split('=>')
    rule_head = rule_head[5:-4]

    mark = ''

    if '?h' in rule_body:
        mark = '?h'

    if '?g' in rule_body:
        mark = '?g'

    if '?f' in rule_body:
        mark = '?f'

    if '?e' in rule_body:
        mark = '?e'

    if '?f' in rule_body:
        mark = '?f'


    if len(mark) > 0:
        r1 = ''
        r2 = ''
        rule_body_list = rule_body.split('  ')[:-1]

        ni_r1_qian = rule_body_list[0] == mark and rule_body_list[2] == '?a'
        shun_r1_qian = rule_body_list[0] == '?a' and rule_body_list[2] == mark
        ni_r1_hou =rule_body_list[3] == mark and rule_body_list[5] == '?a'
        shun_r1_hou = rule_body_list[3] == '?a' and rule_body_list[5] == mark

        ni_r2_qian = rule_body_list[0] == '?b' and rule_body_list[2] == mark
        shun_r2_qian = rule_body_list[0] == mark and rule_body_list[2] == '?b'
        ni_r2_hou = rule_body_list[3] == '?b' and rule_body_list[5] == mark
        shun_r2_hou = rule_body_list[3] == mark and rule_body_list[5] == '?b'

        if ni_r2_qian and shun_r1_hou:
            r2 = '-' + rule_body_list[1]
            r1 = rule_body_list[4]
        elif shun_r2_qian and shun_r1_hou:
            r2 = rule_body_list[1]
            r1 = rule_body_list[4]
        elif ni_r2_hou and shun_r1_qian:
            r2 = '-' + rule_body_list[4]
            r1 = rule_body_list[1]
        elif shun_r2_hou and shun_r1_qian:
            r2 = rule_body_list[4]
            r1 = rule_body_list[1]
        elif ni_r1_hou and ni_r2_qian:
            r1 = '-' + rule_body_list[4]
            r2 = '-' + rule_body_list[1]
        elif ni_r1_hou and shun_r2_qian:
            r1 = '-' + rule_body_list[4]
            r2 = rule_body_list[1]
        elif ni_r1_qian and ni_r2_hou:
            r2 = '-' + rule_body_list[4]
            r1 = '-' + rule_body_list[1]
        elif ni_r1_qian and shun_r2_hou:
            r1 = '-' + rule_body_list[1]
            r2 = rule_body_list[4]

        return r1 + ',' + r2 + ',' + rule_head

    else:
        rule_body_list = rule_body.split('  ')[:-1]
        if len(rule_body.split('  ')) >= 7:
            ni__qian = rule_body_list[0] == '?b' and rule_body_list[2] == '?a'
            shun__qian = rule_body_list[0] == '?a' and rule_body_list[2] == '?b'
            ni__hou = rule_body_list[3] == '?b' and rule_body_list[5] == '?a'
            shun__hou = rule_body_list[3] == '?a' and rule_body_list[5] == '?b'
            if ni__qian and shun__hou:
                r2 = rule_body_list[1]
                r1 = rule_body_list[4]
            elif shun__qian and shun__hou:
                r1 = rule_body_list[1]
                r2 = '-' + rule_body_list[4]
            elif ni__hou and shun__qian:
                r2 = rule_body_list[4]
                r1 = rule_body_list[1]
            elif ni__hou and ni__qian:
                r2 = rule_body_list[4]
                r1 = '-' + rule_body_list[1]
            return r1 + ',' + r2 + ',' + rule_head
        else:
            if rule_body[:2] == '?a':
                return rule_body[4:-7] + ',' + rule_head
            elif rule_body[:2] == '?b':
                return '-' + rule_body[4:-7] + ',' + rule_head




if __name__ == '__main__':

    rule_file = 'rule.txt'
    rule_file2 = 'Rules.txt'

    with open(rule_file, 'r') as f:
        with open(rule_file2, 'w') as f2:
            for line in f:
                Rule, Head_Coverage, Std_Confidence, PCA_Confidence, Positive_Examples, Body_size, PCA_Body_size, _ \
                    = line.strip().split('\t')

                # Rule, PCA_Confidence  = line.strip().split('\t')

                rule_str = rule_transform(Rule)
                newline = rule_str + '\t' + str(PCA_Confidence) + '\n'

                f2.write(newline)
