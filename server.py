#!/usr/bin/python
import SimpleHTTPServer
import SocketServer
import urlparse

class MyRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
	def do_GET(self):
		if self.path.startswith('/report.html'):
			f = open('report.html','r')
			data = f.read()
			query = urlparse.parse_qs(urlparse.urlparse(self.path).query)
			original_id = 1
			if 'id' in query:
				try:
					original_id = int(query['id'][0])
				except ValueError:
					pass
			original_path = '/%s/image_%04d.jpg' % ('caffe_util/101_ObjectCategories/chair',original_id)
			http_response = data % (original_path)

			self.send_response(200,'')
			self.send_header('Content-Length',len(http_response))
			self.send_header('Content-type','text/html')
			self.end_headers()
			self.request.sendall(http_response)
		else:	
			SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

server = SocketServer.TCPServer(('0.0.0.0', 8888), MyRequestHandler)
server.serve_forever()
