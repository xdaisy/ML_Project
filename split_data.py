import sys
def split(file, train):
    all_file = open(file)
    all_text = all_file.read().split("\n")
    all_len = len(all_text)
    train_file = open("tfidf_train.txt", "a")
    train_file.seek(0)
    train_file.truncate()
    for i in range(int(all_len * (train/100.0))):
        train_file.write(all_text[i])
        train_file.write("\n")
    test_file = open("tfidf_test.txt", "a")
    test_file.seek(0)
    test_file.truncate()
    for i in range(int(all_len * (train/100.0)), all_len):
        test_file.write(all_text[i])
        test_file.write("\n")

if __name__ == "__main__":
    split(sys.argv[1], float(sys.argv[2]))