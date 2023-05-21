import pandas as pd

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        if not line.startswith('#'):
            fields = line.strip().split('\t')
            data.append(fields)

    return data

def save_to_csv(data, output_file):
    headers = ['Spanish', 'English', 'Pronunciation', 'Language', 'Category']
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(output_file, index=False, encoding='utf-8')

def main():
    input_file = 'rawData.txt'
    output_file = 'output_data.csv'

    data = read_data(input_file)
    save_to_csv(data, output_file)
    print(f"Data successfully saved to {output_file}")

if __name__ == '__main__':
    main()
