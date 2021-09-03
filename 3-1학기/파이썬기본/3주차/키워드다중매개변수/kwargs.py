def print_keyword_args(**kwargs):
	for keyword, value in kwargs.items():
		print( '{0} = {1}'.format(keyword, value))

print_keyword_args(cow = '누렁이', dog = '멍멍이', cat = '야옹이')
