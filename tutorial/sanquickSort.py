

def quicksort(myList):
    
    if len(myList) <=1:
        return myList 
        
    oList = []
    pivot = myList[0]
    lowerList = []
    upperList = []
    
    for elem in myList[1:]:
        if elem > pivot:
            upperList.append(elem)
        else:
            lowerList.append(elem)
            
    oList = quicksort(lowerList) + [pivot] + quicksort(upperList)
    
    return oList
# Main Method    
if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    myList = [2,3,4,5,9,7,1,6,8]
    print 'List', myList, ' is now order:', quicksort(myList)