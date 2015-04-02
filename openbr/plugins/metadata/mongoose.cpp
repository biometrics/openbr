#include <openbr/plugins/openbr_internal.h>
#include <mongoose.h>

namespace br
{

// This function will be called by mongoose on every new request.
static int begin_request_handler(struct mg_connection *conn) {
  const struct mg_request_info *request_info = mg_get_request_info(conn);
  char content[100];

  // Prepare the message we're going to send
  int content_length = snprintf(content, sizeof(content),
                                "Hello from mongoose! Remote port: %d",
                                request_info->remote_port);

  // Send HTTP reply to the client
  mg_printf(conn,
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: %d\r\n"        // Always set Content-Length
            "\r\n"
            "%s",
            content_length, content);

  // Returning non-zero tells mongoose that our function has replied to
  // the client, and mongoose should not send client any more data.
  return 1;
}

/*!
 * \ingroup initializers
 * \brief Initialize mongoose server
 * \author Unknown \cite Unknown
 */
class MongooseInitializer : public Initializer
{
    Q_OBJECT

    static struct mg_context *ctx;
    static struct mg_callbacks callbacks;

    void initialize() const
    {
        // List of options. Last element must be NULL.
        const char *options[] = { "listening_ports", "8080", NULL };

        // Prepare callbacks structure. We have only one callback, the rest are NULL.
        memset(&callbacks, 0, sizeof(callbacks));
        callbacks.begin_request = begin_request_handler;

        // Start the web server.
        ctx = mg_start(&callbacks, NULL, options);
    }

    void finalize() const
    {
        // Stop the server.
        mg_stop(ctx);
    }
};

struct mg_context *MongooseInitializer::ctx;
struct mg_callbacks MongooseInitializer::callbacks;

BR_REGISTER(Initializer, MongooseInitializer)

} // namespace br

#include "metadata/mongoose.moc"
