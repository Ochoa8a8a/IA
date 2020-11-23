import pandas as pd
import math


def split_n(rows, column):
    names = list(rows.columns)
    gp = rows.groupby([names[column]])
    return gp


def split_two(rows, column, value):
    names = list(rows.columns)
    bigger = rows[rows[names[column]] >= value]
    smaller = rows[rows[names[column]] < value]
    return bigger, smaller


def entropy(rows):
    # Function to calculate entropy
    gp = rows.groupby(['Survived'])
    try:
        t_count = len(gp.get_group(1))
    except:
        t_count = 0
    try:
        f_count = len(gp.get_group(0))
    except:
        f_count = 0
    total = t_count + f_count
    if t_count != 0:
        t_entropy = t_count / total * math.log(t_count / total, 2)
    else:
        t_entropy = 0
    if f_count != 0:
        f_entropy = f_count / total * math.log(f_count / total, 2)
    else:
        f_entropy = 0
    return (t_entropy + f_entropy) * -1


def find_best_split(rows):
    best_gain = 0
    best_column = 0
    column_value = 0
    sys_entropy = entropy(rows)
    n_features = len(rows.columns)
    n_unique = rows.nunique()
    # Iterates all the columns
    for col in range(1, n_features):
        sub_entropy = 0
        t_entries = len(rows.index)
        # If there are less than 3 unique values in column then split them in groups
        if n_unique[col] <= 3:
            gp = split_n(rows, col)
            for group in gp.groups:
                sub_group = gp.get_group(group)
                g_entries = len(sub_group.index)
                p_entropy = entropy(sub_group)
                sub_entropy += p_entropy * (g_entries / t_entries)
            info_gain = sys_entropy - sub_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_column = col
        # If there are more than 3 unique values in column then split in 2 by value
        else:
            visited = []
            for index, row in rows.iterrows():
                if row[col] == 0:
                    continue
                if row[col] in visited:
                    continue
                visited.append(row[col])
                bigger, smaller = split_two(rows, col, row[col])
                b_entries = len(bigger.index)
                s_entries = len(smaller.index)
                if b_entries == 0 or s_entries == 0:
                    continue
                b_entropy = entropy(bigger)
                s_entropy = entropy(smaller)
                sub_entropy = b_entropy * (b_entries / t_entries) + s_entropy * (s_entries / t_entries)
                info_gain = sys_entropy - sub_entropy
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_column = col
                    column_value = row[col]

    return best_gain, best_column, column_value


def count_class(rows):
    # For Leaf class, saves the number of Trues and False in Node
    gp = split_n(rows, 0)
    res = ""
    try:
        res += "True " + str(len(gp.get_group(1))) + ", "
    except:
        pass
    try:
        res += "False " + str(len(gp.get_group(0)))
    except:
        pass
    return res


class Leaf:
    def __init__(self, rows):
        self.predictions = count_class(rows)


class DecisionNode:
    def __init__(self, choice, value):
        self.choice = choice
        self.value = value
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)


def build_tree(data, node):
    # Get best info gain, column and value
    best_gain, best_column, column_value = find_best_split(data)
    if best_gain == 0:
        leaf = Leaf(data)
        node.add_child(leaf)
        return
    n_unique = data.nunique()

    names = list(data.columns)
    if n_unique[best_column] <= 3:
        gp = split_n(data, best_column)
        for group in gp.groups:
            sub_group = gp.get_group(group)
            if len(sub_group) == len(data):
                # If length is the same as original group then it should be a Leaf node
                leaf = Leaf(sub_group)
                node.add_child(leaf)
                print("leaf-end")
            else:
                # Save sub_group into node children and call build tree recursively
                sub_node = DecisionNode(names[best_column], "==" + str(sub_group.iloc[0][names[best_column]]))
                node.add_child(sub_node)
                build_tree(sub_group, sub_node)
    else:
        # Split the group into subgroups and verify
        bigger, smaller = split_two(data, best_column, column_value)
        if not bigger.empty:
            # If length is the same as original group then it should be a Leaf node
            if len(bigger) == len(data):
                leaf = Leaf(data)
                node.add_child(leaf)
                print("leaf-end")
            else:
                # Save bigger sub_group into node children and call build tree recursively
                b_sub_node = DecisionNode(names[best_column], "<=" + str(column_value))
                build_tree(bigger, b_sub_node)
                node.add_child(b_sub_node)
        if not smaller.empty:
            # If length is the same as original group then it should be a Leaf node
            if len(smaller) == len(data):
                leaf = Leaf(data)
                node.add_child(leaf)
                print("leaf-end")
            else:
                # Save smaller sub_group into node children and call build tree recursively
                s_sub_node = DecisionNode(names[best_column], ">" + str(column_value))
                node.add_child(s_sub_node)
                build_tree(smaller, s_sub_node)
    return


def print_tree(node, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.choice) + " " + str(node.value))

    # Call this function recursively on the branches
    for child in node.children:
        print_tree(child, spacing + "  ")


# def classify(row, node):
#     for child in node.children:
#         if isinstance(child, Leaf):
#             pred = child.predictions.split(' ', 1)[0]
#             print("Leaf reached")
#             if pred == 'True':
#                 return True
#             else:
#                 return False
#         choice = child.choice
#         # if row['choice'] == value:
#         e_str = "row['" + str(choice) + "'] " + str(child.value)
#         print(e_str, row[choice], eval(e_str))
#         if eval(e_str):
#             return classify(row, child)
#     print("No match found")
#     return None


def main():
    df = pd.read_csv('train.csv')
    df.replace(['male', 'female', 'S', 'C', 'Q'], [0, 1, 0, 1, 2], inplace=True)
    root = DecisionNode("Root", 0)
    build_tree(df, root)
    print_tree(root)
    dt = pd.read_csv('test.csv')
    dt.replace(['male', 'female', 'S', 'C', 'Q'], [0, 1, 0, 1, 2], inplace=True)
    results = []
    for index, row in dt.iterrows():
        pred = classify(row, root)
        results.append(pred)
    print("tabla", results)
    dr = pd.DataFrame(results, columns=['Survived'])
    dr.replace(['False', 'True'], [0, 1], inplace=True)
    dr.to_csv('results.csv')


main()
