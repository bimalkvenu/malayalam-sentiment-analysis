from collections import defaultdict

data = [
    "big data is powerful",
    "map reduce works well",
    "big data uses map reduce"
]

def map_function(lines):
    mapped = []
    for line in lines:
        words = line.strip().split()
        for word in words:
            mapped.append((word.lower(), 1))  
    return mapped

def shuffle(mapped_data):
    grouped = defaultdict(list)
    for word, count in mapped_data:
        grouped[word].append(count)
    return grouped

def reduce_function(grouped_data):
    reduced = {}
    for word, counts in grouped_data.items():
        reduced[word] = sum(counts)
    return reduced

mapped = map_function(data)
grouped = shuffle(mapped)
reduced = reduce_function(grouped)

print("Word Count Result:")
for word, count in reduced.items():
    print(f"{word}: {count}")
