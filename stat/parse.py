def readLogIntoList(filename):
    bucket = []
    # read saved file format back into Python list
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            tensor2D = line.strip("Sequence #").strip().split(":")
            d = dict()
            d["seqnum"] = int(tensor2D[0])
            tensor2D = tensor2D[1].strip().strip("[[").strip("]]").replace("'", "").split("], [")
            # temp represents a 2D tensor
            for i, tensor1D in enumerate(tensor2D):
                tensor2D[i] = tensor2D[i].split(",")
                for j, elem in enumerate(tensor2D[i]):
                    tensor2D[i][j] = float(elem.strip())
            d["data"] = tensor2D
            bucket.append(d)
    return bucket

def countSparseElements(bucket, threshold = None):
    thresholdFixed = (threshold != None)
    sparseCount = []
    for e in bucket:
        seqnum = e["seqnum"]
        tensor2D = e["data"]
        if not thresholdFixed:
            threshold = 1 / seqnum * 0.1
            #threshold = 1 / seqnum
        mask = [ [1 if v < threshold else 0 for v in r] for r in tensor2D]
        d = dict()
        d["seqnum"] = seqnum
        d["small"] = sum(map(sum, mask))
        d["large"] = len(tensor2D) * len(tensor2D[0]) - d["small"]
        d["sparsity"] = d["small"] / (d["small"] + d["large"])
        sparseCount.append(d)
    return sparseCount          

def getAverageSparsity(sparseCount):
    sparsitySum = 0.
    for d in sparseCount:
        sparsitySum += d["sparsity"]
    return sparsitySum / len(sparseCount)

def getWeightedSparsity(sparseCount):
    smallSum = 0
    largeSum = 0
    for d in sparseCount:
        smallSum += d["small"]
        largeSum += d["large"]
    return smallSum / (smallSum + largeSum)

def main():
    bucket = readLogIntoList("../scores/raw_score.log")
    sparseCount = countSparseElements(bucket)
    print(sparseCount)
    print(f"* =============== Average matrix sparsity: {round(getAverageSparsity(sparseCount), 4)}")
    print(f"* =============== Weighted sparsity: {round(getWeightedSparsity(sparseCount), 4)}")

if __name__ == "__main__":
    main()
