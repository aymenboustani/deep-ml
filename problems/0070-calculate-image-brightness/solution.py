
def calculate_brightness(img):
	# Write your code here
	if (not img
	or any(len(r) != len(img[0]) for r in img)
	or any(p > 255 or p < 0 for P in img for p in P)):
	    return - 1
	else:
		return sum(p for P in img for p in P) / len(img) / len(img[0])
