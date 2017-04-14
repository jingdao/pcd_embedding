#!/usr/bin/python
import SimpleHTTPServer
import SocketServer
import urlparse
import random
from cgi import parse_header, parse_multipart

class MyRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
	def do_GET(self):
		if self.path.startswith('/random'):
			url = '/report.html?id=%s' % random.randint(1,62)
			self.send_response(302,'')
			self.send_header('Location',url)
			self.end_headers()
		if self.path.startswith('/report.html'):
			f = open('report.html','r')
			data = f.read()
			query = urlparse.parse_qs(urlparse.urlparse(self.path).query)
			if 'id' in query:
				try:
					original_id = int(query['id'][0])
				except ValueError:
					pass
				original_path = '/%s/image_%04d.jpg' % ('caffe_util/101_ObjectCategories/chair',original_id)
			else:
				original_path = '/default.jpg'
			http_response = data % (original_path)

			self.send_response(200,'')
			self.send_header('Content-Length',len(http_response))
			self.send_header('Content-type','text/html')
			self.end_headers()
			self.request.sendall(http_response)
		else:	
			SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
	
	def do_POST(self):
		ctype, pdict = parse_header(self.headers['content-type'])
		if ctype == 'multipart/form-data':
			postvars = parse_multipart(self.rfile, pdict)
			data = postvars['myfile'][0]
			f = open('default.jpg','wb')
			f.write(data)
			f.close()
		self.send_response(302,'')
		self.send_header('Location','/report.html')
		self.end_headers()

server = SocketServer.TCPServer(('0.0.0.0', 8888), MyRequestHandler)
server.serve_forever()
