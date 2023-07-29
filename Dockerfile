FROM aambekar234/base-ml-py3.8:v1.0

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

COPY ./ ./

COPY starter.sh /usr/local/bin/

EXPOSE 3100

ENTRYPOINT ["starter.sh"]


    

