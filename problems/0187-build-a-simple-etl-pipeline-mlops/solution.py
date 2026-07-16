# Implement your function below.

def run_etl(csv_text: str) -> list[tuple[str, float]]:
	"""Run a simple ETL pipeline over CSV text with header user_id,event_type,value.

	Returns a sorted list of (user_id, total_value) for event_type == "purchase".
	"""
	# TODO: implement extract, transform, and load steps
	data = csv_text.split("\n")
	headers = data[0].split(",")
	data = [d.split(",") for d in data[1:]]
	for i in range(len(data)):
		for j in range(len(data[i])):
			data[i][j] = data[i][j].strip()
		if len(data[i]) == 1:
			del data[i]
	users = set([d[0] for d in data[1:]])
	values = {user: 0 for user in users}
	valid = {user : True for user in users}
	for d in data:
		if len(d) > 1 and d[1] == "purchase":
			try:
				values[d[0]] += float(d[2])
			except ValueError:
				continue
	values = dict(sorted(values.items()))
	return [(user,value) for user, value in values.items() if value != 0]