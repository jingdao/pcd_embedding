#!/usr/bin/python
import SimpleHTTPServer
import SocketServer
import urlparse
import random
from cgi import parse_header, parse_multipart
import image_server

class MyRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
	def do_GET(self):
		if self.path.startswith('/random'):
			url = '/report.html?id=%s' % random.randint(0,len(image_server.image_list))
			self.send_response(302,'')
			self.send_header('Location',url)
			self.end_headers()
		elif self.path.startswith('/report.html'):
			f = open('report.html','r')
			data = f.read()
			query = urlparse.parse_qs(urlparse.urlparse(self.path).query)
			original_path = image_server.get_path_from_id(query['id'][0] if 'id' in query else None)
			param = [original_path] + ['']*9
			if 'retrieve' in query:
				top = image_server.get_retrieval_results(original_path,query)
				param[1:len(top)+1] = top
			elif 'bgColor' in query:
				top = image_server.get_query_results(original_path,query)
				param[1:len(top)+1] = top
			http_response = data % tuple(param)

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
