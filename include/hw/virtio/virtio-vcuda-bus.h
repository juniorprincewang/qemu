/*
 * Virtio VCuda modified from Virtio VCuda / Console Support
 *
 * Copyright IBM, Corp. 2008
 * Copyright Red Hat, Inc. 2009, 2010
 *
 * Authors:
 *  Christian Ehrhardt <ehrhardt@linux.vnet.ibm.com>
 *  Amit Shah <amit.shah@redhat.com>
 *
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the COPYING file in the top-level directory.
 *
 */

#ifndef _QEMU_VIRTIO_VCUDA_H
#define _QEMU_VIRTIO_VCUDA_H


#include "hw/qdev.h"
#include "hw/virtio/virtio.h"



struct virtio_vcuda_conf {
    /* Max. number of ports we can have for a virtio-vcuda device */
    uint32_t max_virtvcuda_ports;
};

#define TYPE_VIRTIO_VCUDA_PORT "virtio-vcuda-port"
#define VIRTIO_VCUDA_PORT(obj) \
     OBJECT_CHECK(VirtIOVCudaPort, (obj), TYPE_VIRTIO_VCUDA_PORT)
#define VIRTIO_VCUDA_PORT_CLASS(klass) \
     OBJECT_CLASS_CHECK(VirtIOVCudaPortClass, (klass), TYPE_VIRTIO_VCUDA_PORT)
#define VIRTIO_VCUDA_PORT_GET_CLASS(obj) \
     OBJECT_GET_CLASS(VirtIOVCudaPortClass, (obj), TYPE_VIRTIO_VCUDA_PORT)

typedef struct VirtIOVCuda VirtIOVCuda;
typedef struct VirtIOVCudaBus VirtIOVCudaBus;
typedef struct VirtIOVCudaPort VirtIOVCudaPort;

typedef struct VirtIOVCudaPortClass {
    DeviceClass parent_class;

    /* Is this a device that binds with hvc in the guest? */
    bool is_console;
    /*
     * The per-port (or per-app) realize function that's called when a
     * new device is found on the bus.
     */
    DeviceRealize realize;
    /*
     * Per-port unrealize function that's called when a port gets
     * hot-unplugged or removed.
     */
    DeviceUnrealize unrealize;

    /* Callbacks for guest events */
        /* Guest opened/closed device. */
    void (*set_guest_connected)(VirtIOVCudaPort *port, int guest_connected);

    /* Enable/disable backend for virtio vcuda port */
    void (*enable_backend)(VirtIOVCudaPort *port, bool enable);

        /* Guest is now ready to accept data (virtqueues set up). */
    void (*guest_ready)(VirtIOVCudaPort *port);

    /*
     * Guest has enqueued a buffer for the host to write into.
     * Called each time a buffer is enqueued by the guest;
     * irrespective of whether there already were free buffers the
     * host could have consumed.
     *
     * This is dependent on both the guest and host end being
     * connected.
     */
    void (*guest_writable)(VirtIOVCudaPort *port);

    /*
     * Guest wrote some data to the port. This data is handed over to
     * the app via this callback.  The app can return a size less than
     * 'len'.  In this case, throttling will be enabled for this port.
     */
    ssize_t (*have_data)(VirtIOVCudaPort *port, const uint8_t *buf,
                         ssize_t len);
} VirtIOVCudaPortClass;

/*
 * This is the state that's shared between all the ports.  Some of the
 * state is configurable via command-line options. Some of it can be
 * set by individual devices in their initfn routines. Some of the
 * state is set by the generic qdev device init routine.
 */
struct VirtIOVCudaPort {
    DeviceState dev;

    QTAILQ_ENTRY(VirtIOVCudaPort) next;

    /*
     * This field gives us the virtio device as well as the qdev bus
     * that we are associated with
     */
    VirtIOVCuda *vser;

    VirtQueue *ivq, *ovq;

    /*
     * This name is sent to the guest and exported via sysfs.
     * The guest could create symlinks based on this information.
     * The name is in the reverse fqdn format, like org.qemu.console.0
     */
    char *name;

    /*
     * This id helps identify ports between the guest and the host.
     * The guest sends a "header" with this id with each data packet
     * that it sends and the host can then find out which associated
     * device to send out this data to
     */
    uint32_t id;

    /*
     * This is the elem that we pop from the virtqueue.  A slow
     * backend that consumes guest data (e.g. the file backend for
     * qemu chardevs) can cause the guest to block till all the output
     * is flushed.  This isn't desired, so we keep a note of the last
     * element popped and continue consuming it once the backend
     * becomes writable again.
     */
    VirtQueueElement *elem;

    /*
     * The index and the offset into the iov buffer that was popped in
     * elem above.
     */
    uint32_t iov_idx;
    uint64_t iov_offset;

    /*
     * When unthrottling we use a bottom-half to call flush_queued_data.
     */
    QEMUBH *bh;

    /* Is the corresponding guest device open? */
    bool guest_connected;
    /* Is this device open for IO on the host? */
    bool host_connected;
    /* Do apps not want to receive data? */
    bool throttled;
};

/* The virtio-vcuda bus on top of which the ports will ride as devices */
struct VirtIOVCudaBus {
    BusState qbus;

    /* This is the parent device that provides the bus for ports. */
    VirtIOVCuda *vser;

    /* The maximum number of ports that can ride on top of this bus */
    uint32_t max_nr_ports;
};

typedef struct VirtIOVCudaPostLoad {
    QEMUTimer *timer;
    uint32_t nr_active_ports;
    struct {
        VirtIOVCudaPort *port;
        uint8_t host_connected;
    } *connected;
} VirtIOVCudaPostLoad;

struct VirtIOVCuda {
    VirtIODevice parent_obj;

    VirtQueue *c_ivq, *c_ovq;
    /* Arrays of ivqs and ovqs: one per port */
    VirtQueue **ivqs, **ovqs;

    VirtIOVCudaBus bus;

    QTAILQ_HEAD(, VirtIOVCudaPort) ports;

    QLIST_ENTRY(VirtIOVCuda) next;

    /* bitmap for identifying active ports */
    uint32_t *ports_map;

    struct VirtIOVCudaPostLoad *post_load;

    struct virtio_vcuda_conf vcuda;

    uint64_t host_features;
};

/* Interface to the virtio-vcuda bus */

/*
 * Open a connection to the port
 *   Returns 0 on success (always).
 */
int virtio_vcuda_open(VirtIOVCudaPort *port);

/*
 * Close the connection to the port
 *   Returns 0 on success (always).
 */
int virtio_vcuda_close(VirtIOVCudaPort *port);

/*
 * Send data to Guest
 */
ssize_t virtio_vcuda_write(VirtIOVCudaPort *port, const uint8_t *buf,
                            size_t size);

/*
 * Query whether a guest is ready to receive data.
 */
size_t virtio_vcuda_guest_ready(VirtIOVCudaPort *port);

/*
 * Flow control: Ports can signal to the virtio-vcuda core to stop
 * sending data or re-start sending data, depending on the 'throttle'
 * value here.
 */
void virtio_vcuda_throttle_port(VirtIOVCudaPort *port, bool throttle);

#define TYPE_VIRTIO_VCUDA "virtio-vcuda-device"
#define VIRTIO_VCUDA(obj) \
        OBJECT_CHECK(VirtIOVCuda, (obj), TYPE_VIRTIO_VCUDA)

#endif

