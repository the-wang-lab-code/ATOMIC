from utils.utils import post_to


def om_imaging(
        final_objective: str = None,
):
    """
    Call this function when the user has a specific objective in mind, and the system will try to achieve that objective by taking a series of images and analyzing them

    Args:
        final_objective: a string indicating user's final objective, including all the context information user mentioned in that round of conversation, as complete as possible

    Returns:
        None
    """

    data = {
        'final_objective': final_objective,
    }
    resp = post_to('om', 'om_imaging', data)

    return 'Final objective successfully achieved with following detailed descriptions:\n' + resp.text


__functions__ = [
    om_imaging,
]
